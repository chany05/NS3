#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/finebalancer-kpm-store.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/point-to-point-module.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>

using namespace ns3;
using namespace oran;

NS_LOG_COMPONENT_DEFINE("SingleCellFineBalancerScenario");

class SingleCellFineBalancerLoop : public Object
{
  public:
    void Setup(Ptr<LteEnbNetDevice> enb,
               NetDeviceContainer ueDevs,
               double stepTime,
               double collectingWindow,
               uint32_t initCqiTimerThreshold,
               uint32_t initUlGrantMcs,
               std::string csvFilename)
    {
        m_enb = enb;
        m_ueDevs = ueDevs;
        m_stepTime = stepTime;
        m_collectingWindow = collectingWindow;
        m_csvFilename = csvFilename;
        m_store = FineBalancerKpmStore::Get();
        m_store->Clear();

        m_txPower = m_enb->GetPhy()->GetTxPower();
        m_cqiTimerThreshold = initCqiTimerThreshold;
        m_ulGrantMcs = initUlGrantMcs;

        ApplyCqiTimerThreshold();
        ApplyUlGrantMcs();
    }

    void Start()
    {
        Simulator::Schedule(Seconds(m_stepTime),
                            &SingleCellFineBalancerLoop::ResetPhyCounters,
                            this);
        Simulator::Schedule(Seconds(std::max(0.0, 2.0 * m_stepTime - m_collectingWindow)),
                            &SingleCellFineBalancerLoop::StartCollecting,
                            this);
        Simulator::Schedule(Seconds(2.0 * m_stepTime), &SingleCellFineBalancerLoop::RunStep, this);
    }

    void HandleDlPhyTransmission(std::string context, const PhyTransmissionStatParameters params)
    {
        if (!m_collecting)
        {
            return;
        }
        m_window.txCount++;
        // Count distinct TTIs (1ms slots) that had any DL scheduling (true occupancy metric)
        uint64_t sfKey = static_cast<uint64_t>(params.m_timestamp);
        if (sfKey != m_window.lastSfKey)
        {
            m_window.lastSfKey = sfKey;
            m_window.scheduledSubframes++;
        }
        m_window.dlThroughputMbps +=
            static_cast<double>(params.m_size) * 8.0 / 1024.0 / 1024.0 / m_collectingWindow;
    }

    void Finish()
    {
        m_store->WriteJoinedCsv(m_csvFilename);
    }

  private:
    struct Window
    {
        uint32_t txCount{0};
        uint32_t scheduledSubframes{0};
        uint64_t lastSfKey{UINT64_MAX};
        double dlThroughputMbps{0.0};
        std::array<uint32_t, 29> mcsCount{};
    };

    uint32_t EstimateRbCount(uint8_t mcs, uint16_t tbSizeBytes)
    {
        uint32_t tbSizeBits = static_cast<uint32_t>(tbSizeBytes) * 8;
        for (uint32_t nRb = 1; nRb <= 110; ++nRb)
        {
            if (static_cast<uint32_t>(m_amc.GetDlTbSizeFromMcs(mcs, nRb)) >= tbSizeBits)
            {
                return nRb;
            }
        }
        return 110;
    }

    // EARTH project model: P(W) = 130 + 4.7 * 10^((dBm - 30) / 10)
    double EstimatedPowerW(double txPowerDbm) const
    {
        return 130.0 + 4.7 * std::pow(10.0, (txPowerDbm - 30.0) / 10.0);
    }

    void ApplyCqiTimerThreshold()
    {
        std::string path = "/NodeList/" +
                           std::to_string(m_enb->GetNode()->GetId()) +
                           "/DeviceList/*/ComponentCarrierMap/*/LteEnbMac/FfMacScheduler/CqiTimerThreshold";
        Config::Set(path, UintegerValue(m_cqiTimerThreshold));
    }

    void ApplyUlGrantMcs()
    {
        std::string path = "/NodeList/" +
                           std::to_string(m_enb->GetNode()->GetId()) +
                           "/DeviceList/*/ComponentCarrierMap/*/LteEnbMac/FfMacScheduler/UlGrantMcs";
        Config::Set(path, UintegerValue(m_ulGrantMcs));
    }

    void ResetPhyCounters()
    {
        for (uint32_t i = 0; i < m_ueDevs.GetN(); ++i)
        {
            m_ueDevs.Get(i)->GetObject<LteUeNetDevice>()->GetPhy()->ClearFineBalancerDlThroughput();
        }
    }

    void StartCollecting()
    {
        m_collecting = true;
    }

    void RunStep()
    {
        m_collecting = false;

        // KPMs collected here reflect the effect of the PREVIOUS action
        SnapshotKpms();

        // Log the pending actions paired with the just-collected (post-action) KPMs
        for (const auto& a : m_pendingActions)
        {
            m_store->PutAction(m_step, a.xappName, a.paramName, a.value, m_enb->GetCellId(), 0);
        }

        // Apply new action and store as pending for next step
        ApplyXapp();

        ResetWindow();
        m_step++;
        Simulator::Schedule(Seconds(std::max(0.0, m_stepTime - m_collectingWindow)),
                            &SingleCellFineBalancerLoop::StartCollecting,
                            this);
        Simulator::Schedule(Seconds(m_stepTime), &SingleCellFineBalancerLoop::RunStep, this);
    }

    void SnapshotKpms()
    {
        uint16_t cellId = m_enb->GetCellId();
        double totalCqi = 0.0;
        double totalThroughput = 0.0;
        uint32_t servedUes = 0;
        uint32_t farUes = 0;

        for (uint32_t i = 0; i < m_ueDevs.GetN(); ++i)
        {
            Ptr<LteUeNetDevice> ue = m_ueDevs.Get(i)->GetObject<LteUeNetDevice>();
            totalCqi += ue->GetPhy()->GetFineBalancerAvgCqi();
            totalThroughput += ue->GetPhy()->GetFineBalancerDlThroughput();
            servedUes++;
            if (IsFarUe(ue))
            {
                farUes++;
            }
        }

        // Fraction of subframes in the collection window that had any DL scheduling
        double nSubframesTotal = m_collectingWindow * 1000.0;
        double rbUtil = nSubframesTotal == 0.0
                            ? 0.0
                            : std::min(1.0, m_window.scheduledSubframes / nSubframesTotal);
        double avgCqi = servedUes == 0 ? 0.0 : totalCqi / servedUes;
        double farRatio = servedUes == 0 ? 0.0 : static_cast<double>(farUes) / servedUes;

        m_latestRbUtil = rbUtil;
        m_latestDlThroughputMbps = m_window.dlThroughputMbps;
        m_latestAvgCqi = avgCqi;

        // Current parameter state (Param_ prefix → separated into param columns in CSV)
        m_store->PutKpm(m_step, cellId, 0, 0, "Param_TxPower", m_txPower);
        m_store->PutKpm(m_step, cellId, 0, 0, "Param_CqiTimerThreshold", m_cqiTimerThreshold);
        m_store->PutKpm(m_step, cellId, 0, 0, "Param_UlGrantMcs", m_ulGrantMcs);

        // KPMs
        m_store->PutKpm(m_step, cellId, 0, 0, "rbUtil", rbUtil);
        m_store->PutKpm(m_step, cellId, 0, 0, "dlThroughput", m_window.dlThroughputMbps);
        m_store->PutKpm(m_step, cellId, 0, 0, "AvgCqi", avgCqi);
        m_store->PutKpm(m_step, cellId, 0, 0, "EstimatedPower_W", EstimatedPowerW(m_txPower));
        m_store->PutKpm(m_step, cellId, 0, 0, "ServedUes", servedUes);
        m_store->PutKpm(m_step, cellId, 0, 0, "FarUes", farRatio);
    }

    bool IsFarUe(Ptr<LteUeNetDevice> ue) const
    {
        Vector enbPos = m_enb->GetNode()->GetObject<MobilityModel>()->GetPosition();
        Vector uePos = ue->GetNode()->GetObject<MobilityModel>()->GetPosition();
        double dx = enbPos.x - uePos.x;
        double dy = enbPos.y - uePos.y;
        double dz = enbPos.z - uePos.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz) >= 1200.0;
    }

    void ApplyXapp()
    {
        m_pendingActions.clear();
        switch (m_step % 4)
        {
        case 0: {
            // CoverageXapp: raise TxPower when CQI is below target
            if (m_latestAvgCqi < 7.0)
            {
                m_txPower = std::min(46.0, m_txPower + 2.0);
                m_enb->GetPhy()->SetTxPower(m_txPower);
            }
            m_pendingActions.push_back({"CoverageXapp", "TxPower", m_txPower});
            break;
        }
        case 1: {
            // EnergyXapp: TxPower only — reduce when any throughput is delivered (save energy)
            if (m_latestDlThroughputMbps > 0.5)
            {
                m_txPower = std::max(20.0, m_txPower - 2.0);
                m_enb->GetPhy()->SetTxPower(m_txPower);
            }
            m_pendingActions.push_back({"EnergyXapp", "TxPower", m_txPower});
            break;
        }
        case 2: {
            // QoSXapp: CqiTimerThreshold only — sawtooth sweep 1↔10 for observable variation
            if (m_cqiSweepUp)
            {
                m_cqiTimerThreshold = std::min(10u, m_cqiTimerThreshold + 2);
                if (m_cqiTimerThreshold >= 10)
                {
                    m_cqiSweepUp = false;
                }
            }
            else
            {
                m_cqiTimerThreshold = std::max(1u, m_cqiTimerThreshold - 2);
                if (m_cqiTimerThreshold <= 1)
                {
                    m_cqiSweepUp = true;
                }
            }
            ApplyCqiTimerThreshold();
            m_pendingActions.push_back(
                {"QoSXapp", "CqiTimerThreshold", static_cast<double>(m_cqiTimerThreshold)});
            break;
        }
        case 3: {
            // LoadXapp: UlGrantMcs — increase when RB utilization is high
            if (m_latestRbUtil > 0.3)
            {
                m_ulGrantMcs = std::min(28u, m_ulGrantMcs + 1);
                ApplyUlGrantMcs();
            }
            else
            {
                m_ulGrantMcs = std::max(1u, m_ulGrantMcs - 1);
                ApplyUlGrantMcs();
            }
            m_pendingActions.push_back({"LoadXapp", "UlGrantMcs", static_cast<double>(m_ulGrantMcs)});
            break;
        }
        }
    }

    void ResetWindow()
    {
        m_window = Window();
        for (uint32_t i = 0; i < m_ueDevs.GetN(); ++i)
        {
            m_ueDevs.Get(i)->GetObject<LteUeNetDevice>()->GetPhy()->ClearFineBalancerDlThroughput();
        }
    }

    Ptr<LteEnbNetDevice> m_enb;
    NetDeviceContainer m_ueDevs;
    Ptr<FineBalancerKpmStore> m_store;
    LteAmc m_amc;
    Window m_window;
    bool m_collecting{false};
    uint32_t m_step{0};
    double m_stepTime{1.0};
    double m_collectingWindow{0.05};

    double m_txPower{40.0};
    uint32_t m_cqiTimerThreshold{5};
    uint32_t m_ulGrantMcs{14};
    bool m_cqiSweepUp{true};

    double m_latestRbUtil{0.0};
    double m_latestDlThroughputMbps{0.0};
    double m_latestAvgCqi{0.0};

    struct PendingAction
    {
        std::string xappName;
        std::string paramName;
        double value{0.0};
    };
    std::vector<PendingAction> m_pendingActions;

    std::string m_csvFilename{"single-cell-finebalancer-log.csv"};
};

static Ptr<SingleCellFineBalancerLoop> g_loop;

static void
SingleCellDlPhyTransmissionCallback(std::string context, const PhyTransmissionStatParameters params)
{
    if (g_loop)
    {
        g_loop->HandleDlPhyTransmission(context, params);
    }
}

int
main(int argc, char* argv[])
{
    uint32_t numberOfUes = 6;
    double simTime = 20.0;
    double stepTime = 1.0;
    double collectingWindow = 0.05;
    double enbTxPowerDbm = 40.0;
    uint32_t initCqiTimerThreshold = 5;
    uint32_t initUlGrantMcs = 14;
    std::string csvFilename = "single-cell-finebalancer-log.csv";

    CommandLine cmd(__FILE__);
    cmd.AddValue("numberOfUes", "Number of fixed UEs", numberOfUes);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("stepTime", "xApp step interval in seconds", stepTime);
    cmd.AddValue("collectingWindow", "KPM collection window before each step in seconds", collectingWindow);
    cmd.AddValue("enbTxPowerDbm", "Initial eNB TxPower in dBm", enbTxPowerDbm);
    cmd.AddValue("initCqiTimerThreshold", "Initial CqiTimerThreshold (TTI)", initCqiTimerThreshold);
    cmd.AddValue("initUlGrantMcs", "Initial UlGrantMcs (0-28)", initUlGrantMcs);
    cmd.AddValue("csv", "Output CSV filename", csvFilename);
    cmd.Parse(argc, argv);

    Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(true));
    Config::SetDefault("ns3::LteEnbRrc::DefaultTransmissionMode", UintegerValue(2));
    Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(enbTxPowerDbm));
    Config::SetDefault("ns3::UdpClient::Interval", TimeValue(MilliSeconds(20)));
    Config::SetDefault("ns3::UdpClient::PacketSize", UintegerValue(1400));
    Config::SetDefault("ns3::UdpClient::MaxPackets", UintegerValue(10000000));

    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);
    lteHelper->SetSchedulerType("ns3::RrFfMacScheduler");
    lteHelper->SetAttribute("PathlossModel",
                            StringValue("ns3::OkumuraHataPropagationLossModel"));

    Ptr<Node> pgw = epcHelper->GetPgwNode();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);

    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
    p2ph.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    ipv4h.Assign(internetDevices);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    NodeContainer enbNodes;
    NodeContainer ueNodes;
    enbNodes.Create(1);
    ueNodes.Create(numberOfUes);

    MobilityHelper mobility;
    Ptr<ListPositionAllocator> enbPositions = CreateObject<ListPositionAllocator>();
    enbPositions->Add(Vector(0.0, 0.0, 30.0));
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.SetPositionAllocator(enbPositions);
    mobility.Install(enbNodes);

    Ptr<ListPositionAllocator> uePositions = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < numberOfUes; ++i)
    {
        double angle = 2.0 * M_PI * static_cast<double>(i) / std::max(1u, numberOfUes);
        double radius = 500.0 + 375.0 * static_cast<double>(i % 5);
        uePositions->Add(Vector(radius * std::cos(angle), radius * std::sin(angle), 1.5));
    }
    mobility.SetPositionAllocator(uePositions);
    mobility.Install(ueNodes);

    NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);
    Ptr<LteEnbNetDevice> enb = enbLteDevs.Get(0)->GetObject<LteEnbNetDevice>();

    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIfaces = epcHelper->AssignUeIpv4Address(ueLteDevs);

    for (uint32_t i = 0; i < numberOfUes; ++i)
    {
        lteHelper->Attach(ueLteDevs.Get(i), enbLteDevs.Get(0));
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNodes.Get(i)->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    uint16_t dlPort = 10000;
    for (uint32_t i = 0; i < numberOfUes; ++i)
    {
        ++dlPort;
        UdpClientHelper dlClientHelper(ueIpIfaces.GetAddress(i), dlPort);
        PacketSinkHelper dlPacketSinkHelper("ns3::UdpSocketFactory",
                                            InetSocketAddress(Ipv4Address::GetAny(), dlPort));
        ApplicationContainer serverApps = dlPacketSinkHelper.Install(ueNodes.Get(i));
        ApplicationContainer clientApps = dlClientHelper.Install(remoteHost);
        serverApps.Start(Seconds(0.2));
        clientApps.Start(Seconds(0.3));
    }

    g_loop = CreateObject<SingleCellFineBalancerLoop>();
    g_loop->Setup(enb,
                  ueLteDevs,
                  stepTime,
                  collectingWindow,
                  initCqiTimerThreshold,
                  initUlGrantMcs,
                  csvFilename);
    Config::Connect("/NodeList/*/DeviceList/*/ComponentCarrierMap/*/LteEnbPhy/DlPhyTransmission",
                    MakeCallback(&SingleCellDlPhyTransmissionCallback));
    g_loop->Start();

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    g_loop->Finish();
    Simulator::Destroy();
    return 0;
}
