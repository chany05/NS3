#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/finebalancer-kpm-store.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/three-gpp-antenna-model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>

using namespace ns3;
using namespace oran;

NS_LOG_COMPONENT_DEFINE("TwoCellConflictScenario");

// ── Forward declaration ────────────────────────────────────────────────────────
class TwoCellConflictLoop;
static Ptr<TwoCellConflictLoop> g_loop;

// ─────────────────────────────────────────────────────────────────────────────
// TwoCellConflictLoop
//
// Two-cell scenario with three xApps (CCO, ES, MLB).
// All xApps act via direct ns-3 attribute / method calls — no E2 subscription
// or RC procedures.
//
// 6-slot round-robin: each step acts on exactly one (xApp, cell) pair so the
// KPM collection window following each action captures the per-cell effect
// before the next cell is adjusted. This separates cross-cell interference
// from self-interference in the recorded time series.
//   slot = m_step % 6  ->  cell = m_step % 2,  xApp = (m_step/2) % 3
//
// eNB0: (-300, 0, 30m)    eNB1: (300, 0, 30m)
// UEs : clustered near midpoint (0, *, 1.5m) — equidistant from both cells
//
// ICPs  : TxPower, RET, CIO proxy (per cell)
// KPIs  : SINR proxy, dlThroughput, trafficLoad, rbUtil, RSRP, RSRQ,
//         hoSuccessRate, hoFailureRate, hoAttempts
// ─────────────────────────────────────────────────────────────────────────────
class TwoCellConflictLoop : public Object
{
  public:
    // ── Per-cell state ───────────────────────────────────────────────────────
    struct CellState
    {
        Ptr<LteEnbNetDevice> enb;
        Ptr<ThreeGppAntennaModel> antenna;
        uint16_t cellId{0};

        // Controlled parameters (ICP)
        double txPower{40.0};   // dBm  [20, 46]
        double retTiltDeg{8.0}; // deg  [0, 15], positive = electrical downtilt
        double cioProxy{3.0};   // dB   [0, 15], logical CIO control proxy

        // Window accumulators (reset each step)
        uint64_t lastDlSfKey{UINT64_MAX};
        uint32_t scheduledDlSubframes{0};
        double dlThroughputMbps{0.0};
        double sumRsrp{0.0};
        double sumRsrq{0.0};
        uint32_t countRsrp{0};
        double sumUlSinrDb{0.0};
        uint32_t countUlSinr{0};

        // Latest snapshot values used by xApps
        double latestRbUtil{0.5};
        double latestDlThroughputMbps{0.0};
        double latestAvgCqi{7.0};
        double latestDlSinrDb{0.0};
        double latestTrafficLoad{0.0};
        double latestAvgRsrp{-80.0};
        double latestAvgRsrq{-10.0};
        double latestUlSinrDb{10.0};

        void ResetAccumulators()
        {
            lastDlSfKey = UINT64_MAX;
            scheduledDlSubframes = 0;
            dlThroughputMbps = 0.0;
            sumRsrp = 0.0;
            sumRsrq = 0.0;
            countRsrp = 0;
            sumUlSinrDb = 0.0;
            countUlSinr = 0;
        }
    };

    // ── Setup ────────────────────────────────────────────────────────────────
    void Setup(std::array<Ptr<LteEnbNetDevice>, 2> enbs,
               NetDeviceContainer ueDevs,
               double stepTime,
               double collectingWindow,
               double initTxPowerC0,
               double initTxPowerC1,
               double initRetTiltDeg,
               double initCio,
               double mlbImbalanceThreshold,
               double retStepDeg,
               double cioStep,
               std::string controlMode,
               std::string csvFilename)
    {
        m_ueDevs = ueDevs;
        m_stepTime = stepTime;
        m_collectingWindow = collectingWindow;
        m_mlbImbalanceThreshold = mlbImbalanceThreshold;
        m_retStepDeg = retStepDeg;
        m_cioStep = cioStep;
        m_retBaselineDeg = initRetTiltDeg;
        m_cioBaseline = initCio;
        m_controlMode = controlMode;
        m_csvFilename = csvFilename;
        m_store = FineBalancerKpmStore::Get();
        m_store->Clear();

        std::array<double, 2> initTxPower = {initTxPowerC0, initTxPowerC1};
        m_txPowerBaseline = initTxPower;

        for (int i = 0; i < 2; ++i)
        {
            m_cells[i].enb = enbs[i];
            m_cells[i].cellId = enbs[i]->GetCellId();
            m_cells[i].txPower = initTxPower[i];
            m_cells[i].retTiltDeg = initRetTiltDeg;
            m_cells[i].cioProxy = initCio;
            m_cellIdToIdx[m_cells[i].cellId] = i;

            Ptr<Object> antennaObj = enbs[i]->GetPhy()->GetDlSpectrumPhy()->GetAntenna();
            m_cells[i].antenna = antennaObj ? antennaObj->GetObject<ThreeGppAntennaModel>() : nullptr;
            if (m_cells[i].antenna)
            {
                double bearingDeg = (i == 0) ? 0.0 : 180.0;
                m_cells[i].antenna->SetAttribute("BearingAngle", DoubleValue(bearingDeg));
            }
            ApplyParams(m_cells[i]);
        }
    }

    void Start()
    {
        Simulator::Schedule(Seconds(m_stepTime),
                            &TwoCellConflictLoop::ResetPhyCounters,
                            this);
        Simulator::Schedule(Seconds(std::max(0.0, 2.0 * m_stepTime - m_collectingWindow)),
                            &TwoCellConflictLoop::StartCollecting,
                            this);
        Simulator::Schedule(Seconds(2.0 * m_stepTime),
                            &TwoCellConflictLoop::RunStep,
                            this);
    }

    // ── KPM trace callbacks (public — forwarded from free functions) ─────────

    void HandleDlPhyTransmission(const std::string& /*ctx*/,
                                  const PhyTransmissionStatParameters& params)
    {
        if (!m_collecting)
        {
            return;
        }
        auto it = m_cellIdToIdx.find(params.m_cellId);
        if (it == m_cellIdToIdx.end())
        {
            return;
        }
        CellState& cs = m_cells[it->second];
        uint64_t sfKey = static_cast<uint64_t>(params.m_timestamp);
        if (sfKey != cs.lastDlSfKey)
        {
            cs.lastDlSfKey = sfKey;
            cs.scheduledDlSubframes++;
        }
        cs.dlThroughputMbps +=
            static_cast<double>(params.m_size) * 8.0 / 1024.0 / 1024.0 / m_collectingWindow;
    }

    void HandleUeMeasurements(const std::string& /*ctx*/,
                               uint16_t /*rnti*/,
                               uint16_t cellId,
                               double rsrp,
                               double rsrq,
                               bool isServingCell,
                               uint8_t /*ccId*/)
    {
        if (!m_collecting || !isServingCell)
        {
            return;
        }
        auto it = m_cellIdToIdx.find(cellId);
        if (it == m_cellIdToIdx.end())
        {
            return;
        }
        CellState& cs = m_cells[it->second];
        cs.sumRsrp += rsrp;
        cs.sumRsrq += rsrq;
        cs.countRsrp++;
    }

    void HandleUlSinrReport(const std::string& /*ctx*/,
                             uint16_t cellId,
                             uint16_t /*rnti*/,
                             double sinrLinear,
                             uint8_t /*ccId*/)
    {
        if (!m_collecting)
        {
            return;
        }
        auto it = m_cellIdToIdx.find(cellId);
        if (it == m_cellIdToIdx.end())
        {
            return;
        }
        m_cells[it->second].sumUlSinrDb += 10.0 * std::log10(sinrLinear);
        m_cells[it->second].countUlSinr++;
    }

    // ── Handover callbacks (network-level tracking) ──────────────────────────

    void HandleHoStart(uint64_t /*imsi*/,
                        uint16_t /*cellId*/,
                        uint16_t /*rnti*/,
                        uint16_t /*targetCid*/)
    {
        m_stepHoAttempts++;
    }

    void HandleHoEndOk(uint64_t /*imsi*/, uint16_t /*cellId*/, uint16_t /*rnti*/)
    {
        m_stepHoSuccesses++;
    }

    void HandleHoFailure(uint64_t /*imsi*/, uint16_t /*rnti*/, uint16_t /*cellId*/)
    {
        m_stepHoFailures++;
    }

    void Finish()
    {
        m_store->WriteJoinedCsv(m_csvFilename);
    }

  private:
    // ── Parameter application (direct ns-3 calls — no E2/RC procedures) ─────
    void ApplyParams(CellState& cs)
    {
        cs.enb->GetPhy()->SetTxPower(cs.txPower);
        if (cs.antenna)
        {
            cs.antenna->SetAttribute("DowntiltAngle", DoubleValue(cs.retTiltDeg));
        }
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
        RecordParams();
        if (m_controlMode == "profile")
        {
            ApplySingleIcpProfileStep();
        }
        else
        {
            ApplyAllXapps();
        }

        for (int i = 0; i < 2; ++i)
        {
            m_cells[i].ResetAccumulators();
        }
        m_stepHoAttempts = 0;
        m_stepHoSuccesses = 0;
        m_stepHoFailures = 0;

        m_step++;
        Simulator::Schedule(Seconds(std::max(0.0, m_stepTime - m_collectingWindow)),
                            &TwoCellConflictLoop::StartCollecting,
                            this);
        Simulator::Schedule(Seconds(m_stepTime), &TwoCellConflictLoop::RunStep, this);
    }

    // ── KPM snapshot ────────────────────────────────────────────────────────
    // KPM/Param names are prefixed with "c0_" or "c1_" so both cells' values
    // coexist in the same CSV row without overwriting each other.
    void SnapshotKpms()
    {
        // Network-level HO rates (shared across cells)
        double hoSuccessRate =
            m_stepHoAttempts > 0
                ? static_cast<double>(m_stepHoSuccesses) / m_stepHoAttempts
                : 1.0;
        double hoFailureRate =
            m_stepHoAttempts > 0
                ? static_cast<double>(m_stepHoFailures) / m_stepHoAttempts
                : 0.0;
        m_latestHoSuccessRate = hoSuccessRate;
        m_latestHoFailureRate = hoFailureRate;

        double totalDlThroughputMbps = 0.0;
        for (int i = 0; i < 2; ++i)
        {
            m_cells[i].latestDlThroughputMbps = m_cells[i].dlThroughputMbps;
            totalDlThroughputMbps += m_cells[i].latestDlThroughputMbps;
        }

        for (int i = 0; i < 2; ++i)
        {
            CellState& cs = m_cells[i];
            uint16_t cellId = cs.cellId;
            std::string p = "c" + std::to_string(i) + "_"; // column prefix

            cs.latestTrafficLoad = totalDlThroughputMbps > 0.0
                                       ? cs.latestDlThroughputMbps / totalDlThroughputMbps
                                       : 0.5;
            cs.latestRbUtil = std::clamp(cs.latestDlThroughputMbps / 10.0, 0.0, 1.0);

            // Per-cell CQI: iterate UEs and filter by serving cell
            double totalCqi = 0.0;
            uint32_t cqiCount = 0;
            for (uint32_t j = 0; j < m_ueDevs.GetN(); ++j)
            {
                Ptr<LteUeNetDevice> uedev =
                    m_ueDevs.Get(j)->GetObject<LteUeNetDevice>();
                if (uedev->GetRrc()->GetCellId() == cellId)
                {
                    totalCqi += uedev->GetPhy()->GetFineBalancerAvgCqi();
                    cqiCount++;
                }
            }
            cs.latestAvgCqi = cqiCount > 0 ? totalCqi / cqiCount : 7.0;
            cs.latestDlSinrDb =
                DesignedSinrDb(cs.txPower, cs.retTiltDeg, cs.latestTrafficLoad, i);
            cs.latestAvgRsrp =
                cs.countRsrp > 0 ? cs.sumRsrp / cs.countRsrp : -100.0;
            cs.latestAvgRsrq =
                cs.countRsrp > 0 ? cs.sumRsrq / cs.countRsrp : -15.0;
            cs.latestUlSinrDb =
                cs.countUlSinr > 0 ? cs.sumUlSinrDb / cs.countUlSinr : 0.0;

            m_store->PutKpm(m_step, cellId, 0, 0, p + "rbUtil", cs.latestRbUtil);
            m_store->PutKpm(m_step, cellId, 0, 0, p + "dlThroughput", cs.latestDlThroughputMbps);
            m_store->PutKpm(m_step, cellId, 0, 0, p + "AvgCqi", cs.latestAvgCqi);
            m_store->PutKpm(m_step, cellId, 0, 0, p + "SINR", cs.latestDlSinrDb);
            m_store->PutKpm(m_step, cellId, 0, 0, p + "trafficLoad", cs.latestTrafficLoad);
            m_store->PutKpm(m_step, cellId, 0, 0, p + "RSRP", cs.latestAvgRsrp);
            m_store->PutKpm(m_step, cellId, 0, 0, p + "RSRQ", cs.latestAvgRsrq);
            m_store->PutKpm(m_step, cellId, 0, 0, p + "ulSinr", cs.latestUlSinrDb);
            m_store->PutKpm(m_step, cellId, 0, 0, p + "hoSuccessRate", hoSuccessRate);
            m_store->PutKpm(m_step, cellId, 0, 0, p + "hoFailureRate", hoFailureRate);
            m_store->PutKpm(m_step, cellId, 0, 0, p + "hoAttempts",
                            static_cast<double>(m_stepHoAttempts));
        }
    }

    double DesignedSinrDb(double txPowerDbm,
                          double retTiltDeg,
                          double trafficLoad,
                          int cellIndex) const
    {
        double cellOffset = cellIndex == 0 ? 0.3 : -0.3;
        double retCoveragePenalty = 0.08 * retTiltDeg * retTiltDeg;
        return std::clamp(-85.0 + 2.2 * txPowerDbm - retCoveragePenalty - trafficLoad +
                              cellOffset,
                          -10.0,
                          30.0);
    }

    void RecordParams()
    {
        for (int i = 0; i < 2; ++i)
        {
            uint16_t cid = m_cells[i].cellId;
            std::string p = "c" + std::to_string(i) + "_";
            m_store->PutKpm(m_step, cid, 0, 0, p + "Param_TxPower", m_cells[i].txPower);
            m_store->PutKpm(m_step, cid, 0, 0, p + "Param_RET", m_cells[i].retTiltDeg);
            m_store->PutKpm(m_step, cid, 0, 0, p + "Param_CIO", m_cells[i].cioProxy);
        }
    }

    // ── xApp orchestration ───────────────────────────────────────────────────
    //
    // 12-slot round-robin per step: (xApp x cell) pairs are separated so that
    // each step acts on exactly one cell, allowing the KPM collection window
    // after each action to capture the per-cell effect before the next action.
    //
    // Slot  = m_step % 12
    // ES is immediately followed by MLB so ES-induced load shift is visible
    // before the other cell's ES action cancels the imbalance.
    //
    // slot 0: CCO(c0)   slot 1: CCO(c1)
    // slot 2: ES(c0)    slot 3: MLB(c1)
    // slot 4: ES(c1)    slot 5: MLB(c0)
    // slot 6: CCO(c1)   slot 7: CCO(c0)
    // slot 8: ES(c1)    slot 9: MLB(c0)
    // slot10: ES(c0)    slot11: MLB(c1)
    //
    // Only the acted cell's params are pushed to ns-3 after each slot, so the
    // KPMs collected in the next step reflect that single cell's change.
    //
    void ApplyAllXapps()
    {
        int slot = static_cast<int>(m_step % 12);
        int cellIdx = 0;

        switch (slot)
        {
        case 0:
            cellIdx = 0;
            ApplyCCO(cellIdx);
            break;
        case 1:
            cellIdx = 1;
            ApplyCCO(cellIdx);
            break;
        case 2:
            cellIdx = 0;
            ApplyES(cellIdx);
            break;
        case 3:
            cellIdx = 1;
            ApplyMLBCell(cellIdx);
            break;
        case 4:
            cellIdx = 1;
            ApplyES(cellIdx);
            break;
        case 5:
            cellIdx = 0;
            ApplyMLBCell(cellIdx);
            break;
        case 6:
            cellIdx = 1;
            ApplyCCO(cellIdx);
            break;
        case 7:
            cellIdx = 0;
            ApplyCCO(cellIdx);
            break;
        case 8:
            cellIdx = 1;
            ApplyES(cellIdx);
            break;
        case 9:
            cellIdx = 0;
            ApplyMLBCell(cellIdx);
            break;
        case 10:
            cellIdx = 0;
            ApplyES(cellIdx);
            break;
        case 11:
            cellIdx = 1;
            ApplyMLBCell(cellIdx);
            break;
        }

        // Only apply the acted cell — isolates its effect in the next KPM window
        ApplyParams(m_cells[cellIdx]);
    }

    void ApplySingleIcpProfileStep()
    {
        int slot = static_cast<int>(m_step % 6);
        int idx = slot % 2;
        CellState& cs = m_cells[idx];
        std::string cp = "c" + std::to_string(idx) + "_";

        double txpD = 0.0;
        double retD = 0.0;
        double cioD = 0.0;

        switch (slot / 2)
        {
        case 0:
            txpD = ProfileDirection(m_step, 20.0, 46.0, cs.txPower, 2.0);
            cs.txPower = std::clamp(cs.txPower + txpD, 20.0, 46.0);
            m_store->PutAction(m_step, "PROFILE", cp + "TxPower_delta", txpD, cs.cellId, 0);
            break;
        case 1:
            retD = ProfileDirection(m_step, 0.0, 15.0, cs.retTiltDeg, 1.0);
            cs.retTiltDeg = std::clamp(cs.retTiltDeg + retD, 0.0, 15.0);
            m_store->PutAction(m_step, "PROFILE", cp + "RET_delta", retD, cs.cellId, 0);
            break;
        case 2:
            cioD = ProfileDirection(m_step, 0.0, 15.0, cs.cioProxy, 0.5);
            cs.cioProxy = std::clamp(cs.cioProxy + cioD, 0.0, 15.0);
            m_store->PutAction(m_step, "PROFILE", cp + "CIO_delta", cioD, cs.cellId, 0);
            break;
        }

        ApplyParams(cs);
    }

    double ProfileDirection(uint32_t step, double minValue, double maxValue, double current, double delta)
    {
        bool up = ((step / 4) % 2) == 0;
        double requested = up ? delta : -delta;
        if (current + requested > maxValue)
        {
            requested = -delta;
        }
        else if (current + requested < minValue)
        {
            requested = delta;
        }
        return requested;
    }

    // CCO - Coverage & Capacity Optimization
    // KPIs: SINR proxy, dlThroughput
    // ICP : TxPower, then RET as last-priority coverage expansion.
    void ApplyCCO(int idx)
    {
        CellState& cs = m_cells[idx];
        std::string cp = "c" + std::to_string(idx) + "_";
        double txpD = 0.0;
        double retD = 0.0;

        bool coveragePoor = (cs.latestDlSinrDb < 12.0) || (cs.latestDlThroughputMbps < 1.0);
        bool coverageHealthy = (cs.latestDlSinrDb > 12.0) && (cs.latestDlThroughputMbps > 1.5);

        if (coveragePoor)
        {
            if (cs.txPower < 46.0)
            {
                txpD = 2.0;
            }
            else
            {
                retD = -m_retStepDeg;
            }
        }
        else if (coverageHealthy)
        {
            // Restore only one CCO ICP per step so action rows match parameter deltas.
            retD = BaselineDelta(cs.retTiltDeg, m_retBaselineDeg, m_retStepDeg);
            if (std::abs(retD) < 1e-9)
            {
                txpD = BaselineDelta(cs.txPower, m_txPowerBaseline[idx], 1.0);
            }
        }
        cs.txPower = std::clamp(cs.txPower + txpD, 20.0, 46.0);
        cs.retTiltDeg = std::clamp(cs.retTiltDeg + retD, 0.0, 15.0);
        if (std::abs(retD) > 0.0)
        {
            m_store->PutAction(m_step, "CCO", cp + "RET_delta", retD, cs.cellId, 0);
        }
        else
        {
            m_store->PutAction(m_step, "CCO", cp + "TxPower_delta", txpD, cs.cellId, 0);
        }
    }

    // ES - Energy Saving
    // KPIs: AvgCqi, dlThroughput
    // ICP : TxPower
    // Direct conflict with CCO: both control the same TxPower ICP with opposing objectives.
    void ApplyES(int idx)
    {
        CellState& cs = m_cells[idx];
        std::string cp = "c" + std::to_string(idx) + "_";
        double delta = 0.0;
        if (cs.latestDlSinrDb > 15.0 && cs.latestDlThroughputMbps > 1.5)
        {
            delta = -1.0;
        }
        cs.txPower = std::clamp(cs.txPower + delta, 20.0, 46.0);
        m_store->PutAction(m_step, "ES", cp + "TxPower_delta", delta, cs.cellId, 0);
    }

    // MLB - Mobility Load Balancing (per-cell variant)
    // Reads both cells' rbUtil to assess imbalance, but adjusts only cell `idx`.
    // Running MLB(c0) then MLB(c1) in consecutive steps means the KPM window
    // between the two slots captures c0's adjustment effect on both cells before
    // c1 is acted upon, making cross-cell interference visible in the data.
    // ICP: CIO proxy.
    void ApplyMLBCell(int idx)
    {
        double util0    = m_cells[0].latestRbUtil;
        double util1    = m_cells[1].latestRbUtil;
        double imbalance = util0 - util1;

        double cioD = 0.0;

        if (std::abs(imbalance) >= m_mlbImbalanceThreshold)
        {
            int overIdx = (imbalance > 0) ? 0 : 1;

            if (idx == overIdx)
            {
                // This cell is overloaded -> make HO away easier/faster.
                cioD = -m_cioStep;
            }
        }
        else
        {
            cioD = BaselineDelta(m_cells[idx].cioProxy, m_cioBaseline, m_cioStep);
        }

        m_cells[idx].cioProxy = std::clamp(m_cells[idx].cioProxy + cioD, 0.0, 15.0);

        std::string cp = "c" + std::to_string(idx) + "_";
        m_store->PutAction(m_step, "MLB", cp + "CIO_delta", cioD, m_cells[idx].cellId, 0);
    }

    double BaselineDelta(double current, double baseline, double maxStep) const
    {
        double diff = baseline - current;
        if (std::abs(diff) < 1e-9)
        {
            return 0.0;
        }
        return std::clamp(diff, -maxStep, maxStep);
    }

    // ── Member variables ─────────────────────────────────────────────────────
    std::array<CellState, 2> m_cells;
    std::array<double, 2> m_txPowerBaseline{44.0, 44.0};
    std::map<uint16_t, int> m_cellIdToIdx;
    NetDeviceContainer m_ueDevs;
    Ptr<FineBalancerKpmStore> m_store;

    bool m_collecting{false};
    uint32_t m_step{0};
    double m_stepTime{2.0};
    double m_collectingWindow{1.0};
    double m_mlbImbalanceThreshold{0.001};
    double m_retStepDeg{1.0};
    double m_retBaselineDeg{8.0};
    double m_cioStep{1.0};
    double m_cioBaseline{3.0};
    std::string m_controlMode{"closed-loop"};
    std::string m_csvFilename{"two-cell-conflict-log.csv"};

    // Network-level HO counters (reset each step)
    uint32_t m_stepHoAttempts{0};
    uint32_t m_stepHoSuccesses{0};
    uint32_t m_stepHoFailures{0};

    // Latest HO rates kept as KPMs for mobility/load-balancing analysis.
    double m_latestHoSuccessRate{1.0};
    double m_latestHoFailureRate{0.0};
};

// ── Global trace callbacks ────────────────────────────────────────────────────

static void
DlPhyTxCb(std::string ctx, const PhyTransmissionStatParameters p)
{
    if (g_loop)
    {
        g_loop->HandleDlPhyTransmission(ctx, p);
    }
}

static void
UeMeasurementsCb(std::string ctx,
                  uint16_t rnti,
                  uint16_t cellId,
                  double rsrp,
                  double rsrq,
                  bool isServing,
                  uint8_t ccId)
{
    if (g_loop)
    {
        g_loop->HandleUeMeasurements(ctx, rnti, cellId, rsrp, rsrq, isServing, ccId);
    }
}

static void
UlSinrCb(std::string ctx, uint16_t cellId, uint16_t rnti, double sinr, uint8_t ccId)
{
    if (g_loop)
    {
        g_loop->HandleUlSinrReport(ctx, cellId, rnti, sinr, ccId);
    }
}

static void
HoStartCb(std::string /*ctx*/,
           uint64_t imsi,
           uint16_t cellId,
           uint16_t rnti,
           uint16_t targetCid)
{
    if (g_loop)
    {
        g_loop->HandleHoStart(imsi, cellId, rnti, targetCid);
    }
}

static void
HoEndOkCb(std::string /*ctx*/, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    if (g_loop)
    {
        g_loop->HandleHoEndOk(imsi, cellId, rnti);
    }
}

static void
HoFailureCb(std::string /*ctx*/, uint64_t imsi, uint16_t rnti, uint16_t cellId)
{
    if (g_loop)
    {
        g_loop->HandleHoFailure(imsi, rnti, cellId);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int
main(int argc, char* argv[])
{
    uint32_t numberOfUes = 12;
    uint32_t anchorUesPerCell = 1;
    double simTime = 60.0;
    double stepTime = 2.0;
    double collectingWindow = 1.0;
    double enbTxPowerC0Dbm = 44.0;
    double enbTxPowerC1Dbm = 44.0;
    double initRetTiltDeg = 8.0;
    double initCioDb = 3.0;
    double mlbImbalanceThreshold = 0.001;
    double retStepDeg = 1.0;
    double cioStep = 1.0;
    std::string controlMode = "closed-loop";
    std::string csvFilename = "two-cell-conflict-log.csv";

    CommandLine cmd(__FILE__);
    cmd.AddValue("numberOfUes", "Number of boundary UEs used for conflict dynamics", numberOfUes);
    cmd.AddValue("anchorUesPerCell",
                 "Always-attached low-traffic anchor/report UEs per cell",
                 anchorUesPerCell);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("stepTime", "xApp step interval in seconds", stepTime);
    cmd.AddValue("collectingWindow",
                 "KPM collection window before each step in seconds",
                 collectingWindow);
    cmd.AddValue("enbTxPowerC0", "Initial eNB0 TxPower in dBm", enbTxPowerC0Dbm);
    cmd.AddValue("enbTxPowerC1", "Initial eNB1 TxPower in dBm", enbTxPowerC1Dbm);
    cmd.AddValue("initRet", "Initial electrical downtilt/RET angle in degrees", initRetTiltDeg);
    cmd.AddValue("initCio", "Initial CIO control proxy in dB", initCioDb);
    cmd.AddValue("mlbImbalanceThreshold",
                 "RB-utilization imbalance threshold that triggers MLB mobility control",
                 mlbImbalanceThreshold);
    cmd.AddValue("retStep", "Per-action electrical downtilt/RET delta in degrees for CCO", retStepDeg);
    cmd.AddValue("cioStep", "Per-action CIO proxy delta in dB for MLB", cioStep);
    cmd.AddValue("controlMode", "Control mode: closed-loop or profile", controlMode);
    cmd.AddValue("csv", "Output CSV filename", csvFilename);
    cmd.Parse(argc, argv);

    Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(true));
    Config::SetDefault("ns3::LteEnbRrc::DefaultTransmissionMode", UintegerValue(2));
    // Use lower of the two initial powers as the ns3 default (individual cells set later via xApp)
    Config::SetDefault("ns3::LteEnbPhy::TxPower",
                       DoubleValue(std::min(enbTxPowerC0Dbm, enbTxPowerC1Dbm)));
    Config::SetDefault("ns3::UdpClient::Interval", TimeValue(MilliSeconds(20)));
    Config::SetDefault("ns3::UdpClient::PacketSize", UintegerValue(1400));
    Config::SetDefault("ns3::UdpClient::MaxPackets", UintegerValue(10000000));
    Config::SetDefault("ns3::LteUePhy::UeMeasurementsFilterPeriod", TimeValue(MilliSeconds(200)));

    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);
    lteHelper->SetSchedulerType("ns3::RrFfMacScheduler");
    lteHelper->SetAttribute("UsePdschForCqiGeneration", BooleanValue(false));
    lteHelper->SetEnbAntennaModelType("ns3::ThreeGppAntennaModel");
    lteHelper->SetUeAntennaModelType("ns3::IsotropicAntennaModel");

    // Large-scale path loss: OkumuraHata urban macro (for ~200–600 m cells)
    lteHelper->SetAttribute("PathlossModel",
                             StringValue("ns3::OkumuraHataPropagationLossModel"));

    // Fading is intentionally disabled in this validation scenario so the
    // graph reconstruction first sees clean ICP->KPI relationships.
    lteHelper->SetFadingModel("");

    lteHelper->SetHandoverAlgorithmType("ns3::A3RsrpHandoverAlgorithm");
    constexpr double kFixedA3HysteresisDb = 3.0;
    constexpr int64_t kFixedA3TttMs = 128;
    lteHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(kFixedA3HysteresisDb));
    lteHelper->SetHandoverAlgorithmAttribute("TimeToTrigger",
                                              TimeValue(MilliSeconds(kFixedA3TttMs)));

    // ── Core network ──────────────────────────────────────────────────────────
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

    // ── Nodes ─────────────────────────────────────────────────────────────────
    NodeContainer enbNodes;
    NodeContainer ueNodes;
    enbNodes.Create(2);
    uint32_t totalAnchorUes = 2 * anchorUesPerCell;
    uint32_t totalUes = numberOfUes + totalAnchorUes;
    ueNodes.Create(totalUes);

    // ── Mobility ──────────────────────────────────────────────────────────────
    // eNB0: (-200, 0, 30m)   eNB1: (200, 0, 30m)   — 400 m inter-site distance
    // Narrower spacing (vs 600 m) → stronger inter-cell interference at the
    // boundary and greater RSRP sensitivity to TxPower changes (±2 dBm shifts
    // the fixed A3 margin without exposing HYS/TTT as xApp-controlled ICPs).
    //
    // Boundary UE placement:
    //   Boundary UEs are fixed on the perpendicular bisector x=0, so every
    //   boundary UE has exactly the same geometric distance to both eNBs.
    //   Initial boundary attachment is balanced 1:1 instead of forced 3:1.
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    Ptr<ListPositionAllocator> enbPositions = CreateObject<ListPositionAllocator>();
    enbPositions->Add(Vector(-200.0, 0.0, 30.0));
    enbPositions->Add(Vector(200.0, 0.0, 30.0));
    mobility.SetPositionAllocator(enbPositions);
    mobility.Install(enbNodes);

    // UE placement — anchors plus boundary UEs:
    //
    // Anchor UEs sit near their serving eNB and are force-attached there. They
    // keep each cell observable even if all boundary UEs hand over to one side.
    // Boundary UEs remain at the midpoint and drive the conflict dynamics.
    //
    // Boundary UEs are placed on the center line. They stay fixed so KPM changes
    // come from xApp control and fading/interference, not UE mobility.
    Ptr<ListPositionAllocator> uePositions = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < anchorUesPerCell; ++i)
    {
        uePositions->Add(Vector(-180.0, 5.0 * static_cast<double>(i), 1.5));
    }
    for (uint32_t i = 0; i < anchorUesPerCell; ++i)
    {
        uePositions->Add(Vector(180.0, 5.0 * static_cast<double>(i), 1.5));
    }
    for (uint32_t i = 0; i < numberOfUes; ++i)
    {
        double y = 0.0;
        if (numberOfUes > 1)
        {
            y = -60.0 + 120.0 * static_cast<double>(i) / static_cast<double>(numberOfUes - 1);
        }
        uePositions->Add(Vector(0.0, y, 1.5));
    }
    mobility.SetPositionAllocator(uePositions);
    mobility.Install(ueNodes);

    // Initial boundary attach: balanced c0/c1 alternation, not an artificial bias.
    // Anchors are attached before boundary UEs and remain close enough to their
    // serving eNB that normal A3 handover should not move them.

    // ── LTE devices ───────────────────────────────────────────────────────────
    NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);

    lteHelper->AddX2Interface(enbNodes);

    Ptr<LteEnbNetDevice> enb0 = enbLteDevs.Get(0)->GetObject<LteEnbNetDevice>();
    Ptr<LteEnbNetDevice> enb1 = enbLteDevs.Get(1)->GetObject<LteEnbNetDevice>();

    // Apply initial TxPower immediately after device creation.
    enb0->GetPhy()->SetTxPower(enbTxPowerC0Dbm);
    enb1->GetPhy()->SetTxPower(enbTxPowerC1Dbm);

    // ── IP stack and UE attachment ────────────────────────────────────────────
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIfaces = epcHelper->AssignUeIpv4Address(ueLteDevs);

    // Attach anchors first, then attach boundary UEs in a balanced pattern.
    for (uint32_t i = 0; i < totalUes; ++i)
    {
        Ptr<NetDevice> enbDev;
        if (i < anchorUesPerCell)
        {
            enbDev = enbLteDevs.Get(0);
        }
        else if (i < totalAnchorUes)
        {
            enbDev = enbLteDevs.Get(1);
        }
        else
        {
            uint32_t boundaryIdx = i - totalAnchorUes;
            enbDev = (boundaryIdx % 2 == 0) ? enbLteDevs.Get(0) : enbLteDevs.Get(1);
        }
        lteHelper->Attach(ueLteDevs.Get(i), enbDev);

        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNodes.Get(i)->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    // ── DL + UL UDP traffic ───────────────────────────────────────────────────
    Ipv4Address remoteHostAddr("1.0.0.2");
    uint16_t dlPort = 10000;
    uint16_t ulPort = 20000;
    for (uint32_t i = 0; i < totalUes; ++i)
    {
        bool isAnchor = i < totalAnchorUes;

        ++dlPort;
        UdpClientHelper dlClientHelper(ueIpIfaces.GetAddress(i), dlPort);
        dlClientHelper.SetAttribute("Interval",
                                    TimeValue(MilliSeconds(isAnchor ? 100 : 20)));
        dlClientHelper.SetAttribute("PacketSize", UintegerValue(isAnchor ? 300 : 1400));
        PacketSinkHelper dlSinkHelper("ns3::UdpSocketFactory",
                                       InetSocketAddress(Ipv4Address::GetAny(), dlPort));
        dlSinkHelper.Install(ueNodes.Get(i)).Start(Seconds(0.2));
        dlClientHelper.Install(remoteHost).Start(Seconds(0.3));

        ++ulPort;
        UdpClientHelper ulClientHelper(remoteHostAddr, ulPort);
        ulClientHelper.SetAttribute("Interval",
                                    TimeValue(MilliSeconds(isAnchor ? 100 : 20)));
        ulClientHelper.SetAttribute("PacketSize", UintegerValue(isAnchor ? 300 : 1400));
        PacketSinkHelper ulSinkHelper("ns3::UdpSocketFactory",
                                       InetSocketAddress(Ipv4Address::GetAny(), ulPort));
        ulSinkHelper.Install(remoteHost).Start(Seconds(0.2));
        ulClientHelper.Install(ueNodes.Get(i)).Start(Seconds(0.3));
    }

    // ── Conflict loop setup ───────────────────────────────────────────────────
    g_loop = CreateObject<TwoCellConflictLoop>();
    g_loop->Setup({enb0, enb1},
                   ueLteDevs,
                   stepTime,
                   collectingWindow,
                   enbTxPowerC0Dbm,
                   enbTxPowerC1Dbm,
                   initRetTiltDeg,
                   initCioDb,
                   mlbImbalanceThreshold,
                   retStepDeg,
                   cioStep,
                   controlMode,
                   csvFilename);

    // ── Connect KPM traces (direct — no O-RAN subscription) ──────────────────
    Config::Connect("/NodeList/*/DeviceList/*/ComponentCarrierMap/*/LteEnbPhy/DlPhyTransmission",
                    MakeCallback(&DlPhyTxCb));
    Config::Connect(
        "/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/LteUePhy/ReportUeMeasurements",
        MakeCallback(&UeMeasurementsCb));
    Config::Connect("/NodeList/*/DeviceList/*/ComponentCarrierMap/*/LteEnbPhy/ReportUeSinr",
                    MakeCallback(&UlSinrCb));

    // ── Connect HO traces ─────────────────────────────────────────────────────
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverStart",
                    MakeCallback(&HoStartCb));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
                    MakeCallback(&HoEndOkCb));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverFailureNoPreamble",
                    MakeCallback(&HoFailureCb));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverFailureMaxRach",
                    MakeCallback(&HoFailureCb));

    g_loop->Start();

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    g_loop->Finish();
    Simulator::Destroy();
    return 0;
}
