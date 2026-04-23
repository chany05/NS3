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
               double initTxMode2GainDb,
               double initUeTxPowerDbm,
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
        m_txMode2Gain = initTxMode2GainDb;
        m_ueTxPower = initUeTxPowerDbm;

        ApplyTxMode2Gain();
        ApplyUeTxPower();
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

    // ── DL PHY: rbUtil, dlThroughput ────────────────────────────────────────
    void HandleDlPhyTransmission(std::string /*context*/,
                                 const PhyTransmissionStatParameters params)
    {
        if (!m_collecting)
        {
            return;
        }
        uint64_t sfKey = static_cast<uint64_t>(params.m_timestamp);
        if (sfKey != m_window.lastDlSfKey)
        {
            m_window.lastDlSfKey = sfKey;
            m_window.scheduledDlSubframes++;
        }
        m_window.dlThroughputMbps +=
            static_cast<double>(params.m_size) * 8.0 / 1024.0 / 1024.0 / m_collectingWindow;
    }

    // ── UE measurements: RSRP, RSRQ ────────────────────────────────────────
    void HandleUeMeasurements(std::string /*context*/,
                              uint16_t /*rnti*/,
                              uint16_t /*cellId*/,
                              double rsrp,
                              double rsrq,
                              bool isServingCell,
                              uint8_t /*ccId*/)
    {
        if (!m_collecting || !isServingCell)
        {
            return;
        }
        m_window.sumRsrp += rsrp;
        m_window.sumRsrq += rsrq;
        m_window.countRsrp++;
    }

    // ── UL PHY: ulThroughput ────────────────────────────────────────────────
    void HandleUlPhyTransmission(std::string /*context*/,
                                 const PhyTransmissionStatParameters params)
    {
        if (!m_collecting)
        {
            return;
        }
        m_window.ulThroughputMbps +=
            static_cast<double>(params.m_size) * 8.0 / 1024.0 / 1024.0 / m_collectingWindow;
    }

    // ── eNB UL SINR (SRS-based): ulSinr ────────────────────────────────────
    void HandleUlSinrReport(std::string /*context*/,
                            uint16_t /*cellId*/,
                            uint16_t /*rnti*/,
                            double sinrLinear,
                            uint8_t /*ccId*/)
    {
        if (!m_collecting)
        {
            return;
        }
        m_window.sumUlSinrDb += 10.0 * std::log10(sinrLinear);
        m_window.countUlSinr++;
    }

    // ── eNB UL interference spectrum: ulInterference ───────────────────────
    void HandleUlInterference(std::string /*context*/,
                              uint16_t /*cellId*/,
                              Ptr<SpectrumValue> interference)
    {
        if (!m_collecting)
        {
            return;
        }
        m_window.sumUlInterference += Sum(*interference);
        m_window.countUlInterference++;
    }

    void Finish()
    {
        m_store->WriteJoinedCsv(m_csvFilename);
    }

  private:
    struct Window
    {
        // DL
        uint32_t scheduledDlSubframes{0};
        uint64_t lastDlSfKey{UINT64_MAX};
        double dlThroughputMbps{0.0};
        // RSRP/RSRQ
        double sumRsrp{0.0};
        double sumRsrq{0.0};
        uint32_t countRsrp{0};
        // UL throughput
        double ulThroughputMbps{0.0};
        // UL SINR
        double sumUlSinrDb{0.0};
        uint32_t countUlSinr{0};
        // UL interference
        double sumUlInterference{0.0};
        uint32_t countUlInterference{0};
    };

    // EARTH project model: P(W) = 130 + 4.7 * 10^((dBm - 30) / 10)
    double EstimatedPowerW(double txPowerDbm) const
    {
        return 130.0 + 4.7 * std::pow(10.0, (txPowerDbm - 30.0) / 10.0);
    }

    void ApplyTxMode2Gain()
    {
        // TxMode2Gain multiplies the SINR used for DL CQI feedback in mode-2 UEs
        Config::Set("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/LteUePhy/TxMode2Gain",
                    DoubleValue(m_txMode2Gain));
    }

    void ApplyUeTxPower()
    {
        // UE UL transmit power (affects ulThroughput and ulSinr at eNB)
        Config::Set("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/LteUePhy/TxPower",
                    DoubleValue(m_ueTxPower));
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

        SnapshotKpms();

        for (const auto& a : m_pendingActions)
        {
            m_store->PutAction(m_step, a.xappName, a.paramName, a.value, m_enb->GetCellId(), 0);
        }

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
        uint32_t servedUes = 0;
        uint32_t farUes = 0;

        for (uint32_t i = 0; i < m_ueDevs.GetN(); ++i)
        {
            Ptr<LteUeNetDevice> ue = m_ueDevs.Get(i)->GetObject<LteUeNetDevice>();
            totalCqi += ue->GetPhy()->GetFineBalancerAvgCqi();
            servedUes++;
            if (IsFarUe(ue))
            {
                farUes++;
            }
        }

        double nSubframesTotal = m_collectingWindow * 1000.0;
        double rbUtil = nSubframesTotal == 0.0
                            ? 0.0
                            : std::min(1.0, m_window.scheduledDlSubframes / nSubframesTotal);
        double avgCqi = servedUes == 0 ? 0.0 : totalCqi / servedUes;
        double farRatio = servedUes == 0 ? 0.0 : static_cast<double>(farUes) / servedUes;
        double avgRsrp =
            m_window.countRsrp == 0 ? 0.0 : m_window.sumRsrp / m_window.countRsrp;
        double avgRsrq =
            m_window.countRsrp == 0 ? 0.0 : m_window.sumRsrq / m_window.countRsrp;
        double avgUlSinrDb =
            m_window.countUlSinr == 0 ? 0.0 : m_window.sumUlSinrDb / m_window.countUlSinr;
        double avgUlInterference =
            m_window.countUlInterference == 0
                ? 0.0
                : m_window.sumUlInterference / m_window.countUlInterference;

        m_latestRbUtil = rbUtil;
        m_latestDlThroughputMbps = m_window.dlThroughputMbps;
        m_latestAvgCqi = avgCqi;

        // Parameters (Param_ prefix → separated into param columns in CSV)
        m_store->PutKpm(m_step, cellId, 0, 0, "Param_TxPower", m_txPower);
        m_store->PutKpm(m_step, cellId, 0, 0, "Param_TxMode2Gain", m_txMode2Gain);
        m_store->PutKpm(m_step, cellId, 0, 0, "Param_UeTxPower", m_ueTxPower);

        // DL KPMs
        m_store->PutKpm(m_step, cellId, 0, 0, "rbUtil", rbUtil);
        m_store->PutKpm(m_step, cellId, 0, 0, "dlThroughput", m_window.dlThroughputMbps);
        m_store->PutKpm(m_step, cellId, 0, 0, "AvgCqi", avgCqi);
        m_store->PutKpm(m_step, cellId, 0, 0, "EstimatedPower_W", EstimatedPowerW(m_txPower));
        m_store->PutKpm(m_step, cellId, 0, 0, "RSRP", avgRsrp);
        m_store->PutKpm(m_step, cellId, 0, 0, "RSRQ", avgRsrq);

        // UL KPMs
        m_store->PutKpm(m_step, cellId, 0, 0, "ulThroughput", m_window.ulThroughputMbps);
        m_store->PutKpm(m_step, cellId, 0, 0, "ulSinr", avgUlSinrDb);
        m_store->PutKpm(m_step, cellId, 0, 0, "ulInterference", avgUlInterference);

        // Coverage
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
            // CoverageXapp: raise eNB TxPower when DL CQI is below target
            if (m_latestAvgCqi < 7.0)
            {
                m_txPower = std::min(46.0, m_txPower + 2.0);
                m_enb->GetPhy()->SetTxPower(m_txPower);
            }
            m_pendingActions.push_back({"CoverageXapp", "TxPower", m_txPower});
            break;
        }
        case 1: {
            // EnergyXapp: reduce eNB TxPower when throughput is sufficient (save energy)
            // Direct conflict with CoverageXapp on TxPower
            if (m_latestDlThroughputMbps > 0.5)
            {
                m_txPower = std::max(20.0, m_txPower - 2.0);
                m_enb->GetPhy()->SetTxPower(m_txPower);
            }
            m_pendingActions.push_back({"EnergyXapp", "TxPower", m_txPower});
            break;
        }
        case 2: {
            // QoSXapp: sweep TxMode2Gain 0↔4.2 dB at half speed (period=16 steps)
            // Higher gain → higher effective SINR for CQI → scheduler selects higher MCS
            // Independent variation from UeTxPower (different period)
            if (m_step % 16 < 8)
            {
                m_txMode2Gain = std::min(4.2, m_txMode2Gain + 1.05);
            }
            else
            {
                m_txMode2Gain = std::max(0.0, m_txMode2Gain - 1.05);
            }
            ApplyTxMode2Gain();
            m_pendingActions.push_back({"QoSXapp", "TxMode2Gain", m_txMode2Gain});
            break;
        }
        case 3: {
            // LoadXapp: sweep UE UL TxPower 10↔23 dBm at double speed (period=4 steps)
            // Higher UL power → higher ulSinr → higher ulThroughput
            if (m_ueTxPowerUp)
            {
                m_ueTxPower = std::min(23.0, m_ueTxPower + 6.5);
                if (m_ueTxPower >= 23.0)
                {
                    m_ueTxPowerUp = false;
                }
            }
            else
            {
                m_ueTxPower = std::max(10.0, m_ueTxPower - 6.5);
                if (m_ueTxPower <= 10.0)
                {
                    m_ueTxPowerUp = true;
                }
            }
            ApplyUeTxPower();
            m_pendingActions.push_back({"LoadXapp", "UeTxPower", m_ueTxPower});
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
    Window m_window;
    bool m_collecting{false};
    uint32_t m_step{0};
    double m_stepTime{1.0};
    double m_collectingWindow{0.05};

    // Controlled parameters
    double m_txPower{40.0};
    double m_txMode2Gain{4.2};
    double m_ueTxPower{23.0};

    // Sweep direction flags
    bool m_txMode2GainUp{false};
    bool m_ueTxPowerUp{false};

    // Latest KPM snapshot (used by xApps)
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

// ── Global callbacks (ns3 trace system requires free functions) ──────────────

static Ptr<SingleCellFineBalancerLoop> g_loop;

static void
DlPhyTxCb(std::string context, const PhyTransmissionStatParameters p)
{
    if (g_loop)
    {
        g_loop->HandleDlPhyTransmission(context, p);
    }
}

static void
UeMeasurementsCb(std::string context,
                 uint16_t rnti,
                 uint16_t cellId,
                 double rsrp,
                 double rsrq,
                 bool isServing,
                 uint8_t ccId)
{
    if (g_loop)
    {
        g_loop->HandleUeMeasurements(context, rnti, cellId, rsrp, rsrq, isServing, ccId);
    }
}

static void
UlPhyTxCb(std::string context, const PhyTransmissionStatParameters p)
{
    if (g_loop)
    {
        g_loop->HandleUlPhyTransmission(context, p);
    }
}

static void
UlSinrCb(std::string context, uint16_t cellId, uint16_t rnti, double sinr, uint8_t ccId)
{
    if (g_loop)
    {
        g_loop->HandleUlSinrReport(context, cellId, rnti, sinr, ccId);
    }
}

static void
UlInterferenceCb(std::string context, uint16_t cellId, Ptr<SpectrumValue> interf)
{
    if (g_loop)
    {
        g_loop->HandleUlInterference(context, cellId, interf);
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
    double initTxMode2GainDb = 4.2;
    double initUeTxPowerDbm = 23.0;
    std::string csvFilename = "single-cell-finebalancer-log.csv";

    CommandLine cmd(__FILE__);
    cmd.AddValue("numberOfUes", "Number of fixed UEs", numberOfUes);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("stepTime", "xApp step interval in seconds", stepTime);
    cmd.AddValue("collectingWindow", "KPM collection window before each step in seconds",
                 collectingWindow);
    cmd.AddValue("enbTxPowerDbm", "Initial eNB TxPower in dBm", enbTxPowerDbm);
    cmd.AddValue("initTxMode2GainDb", "Initial TxMode2Gain in dB (default 4.2)", initTxMode2GainDb);
    cmd.AddValue("initUeTxPowerDbm", "Initial UE UL TxPower in dBm (default 23)", initUeTxPowerDbm);
    cmd.AddValue("csv", "Output CSV filename", csvFilename);
    cmd.Parse(argc, argv);

    Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(true));
    Config::SetDefault("ns3::LteEnbRrc::DefaultTransmissionMode", UintegerValue(2));
    Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(enbTxPowerDbm));
    Config::SetDefault("ns3::UdpClient::Interval", TimeValue(MilliSeconds(20)));
    Config::SetDefault("ns3::UdpClient::PacketSize", UintegerValue(1400));
    Config::SetDefault("ns3::UdpClient::MaxPackets", UintegerValue(10000000));
    // Report UE measurements every 50ms so they land in the 50ms collection window
    Config::SetDefault("ns3::LteUePhy::UeMeasurementsFilterPeriod",
                       TimeValue(MilliSeconds(50)));

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

    Ipv4InterfaceContainer remoteIfaces = ipv4h.Assign(NetDeviceContainer());
    Ipv4Address remoteHostAddr("1.0.0.2");

    uint16_t dlPort = 10000;
    uint16_t ulPort = 20000;
    for (uint32_t i = 0; i < numberOfUes; ++i)
    {
        // DL: remoteHost → UE
        ++dlPort;
        UdpClientHelper dlClientHelper(ueIpIfaces.GetAddress(i), dlPort);
        PacketSinkHelper dlSinkHelper("ns3::UdpSocketFactory",
                                      InetSocketAddress(Ipv4Address::GetAny(), dlPort));
        dlSinkHelper.Install(ueNodes.Get(i)).Start(Seconds(0.2));
        dlClientHelper.Install(remoteHost).Start(Seconds(0.3));

        // UL: UE → remoteHost
        ++ulPort;
        UdpClientHelper ulClientHelper(remoteHostAddr, ulPort);
        PacketSinkHelper ulSinkHelper("ns3::UdpSocketFactory",
                                      InetSocketAddress(Ipv4Address::GetAny(), ulPort));
        ulSinkHelper.Install(remoteHost).Start(Seconds(0.2));
        ulClientHelper.Install(ueNodes.Get(i)).Start(Seconds(0.3));
    }

    g_loop = CreateObject<SingleCellFineBalancerLoop>();
    g_loop->Setup(enb,
                  ueLteDevs,
                  stepTime,
                  collectingWindow,
                  initTxMode2GainDb,
                  initUeTxPowerDbm,
                  csvFilename);

    // Connect KPM traces directly (no O-RAN subscription — FineBalancer design)
    Config::Connect(
        "/NodeList/*/DeviceList/*/ComponentCarrierMap/*/LteEnbPhy/DlPhyTransmission",
        MakeCallback(&DlPhyTxCb));
    Config::Connect(
        "/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/LteUePhy/ReportUeMeasurements",
        MakeCallback(&UeMeasurementsCb));
    Config::Connect(
        "/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/LteUePhy/UlPhyTransmission",
        MakeCallback(&UlPhyTxCb));
    Config::Connect(
        "/NodeList/*/DeviceList/*/ComponentCarrierMap/*/LteEnbPhy/ReportUeSinr",
        MakeCallback(&UlSinrCb));
    Config::Connect(
        "/NodeList/*/DeviceList/*/ComponentCarrierMap/*/LteEnbPhy/ReportInterference",
        MakeCallback(&UlInterferenceCb));

    g_loop->Start();

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    g_loop->Finish();
    Simulator::Destroy();
    return 0;
}
