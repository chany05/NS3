#include "finebalancer-kpm-store.h"

#include "ns3/log.h"
#include "ns3/simulator.h"

#include <fstream>
#include <map>
#include <set>
#include <vector>

namespace ns3
{
namespace oran
{

NS_LOG_COMPONENT_DEFINE("FineBalancerKpmStore");
NS_OBJECT_ENSURE_REGISTERED(FineBalancerKpmStore);

TypeId
FineBalancerKpmStore::GetTypeId()
{
    static TypeId tid = TypeId("ns3::oran::FineBalancerKpmStore")
                            .SetParent<Object>()
                            .SetGroupName("Oran")
                            .AddConstructor<FineBalancerKpmStore>();
    return tid;
}

Ptr<FineBalancerKpmStore>
FineBalancerKpmStore::Get()
{
    static Ptr<FineBalancerKpmStore> store = CreateObject<FineBalancerKpmStore>();
    return store;
}

void
FineBalancerKpmStore::Clear()
{
    m_kpmSamples.clear();
    m_actionSamples.clear();
}

void
FineBalancerKpmStore::PutKpm(uint32_t step,
                             uint16_t cellId,
                             uint16_t rnti,
                             uint64_t imsi,
                             const std::string& kpmName,
                             double value)
{
    m_kpmSamples.push_back({step, Simulator::Now(), cellId, rnti, imsi, kpmName, value});
}

void
FineBalancerKpmStore::PutAction(uint32_t step,
                                const std::string& xappName,
                                const std::string& adjustedParam,
                                double adjustedValue,
                                uint16_t cellId,
                                uint16_t rnti)
{
    m_actionSamples.push_back(
        {step, Simulator::Now(), xappName, adjustedParam, adjustedValue, cellId, rnti});
}

std::vector<FineBalancerKpmSample>
FineBalancerKpmStore::GetKpmSamples() const
{
    return m_kpmSamples;
}

std::vector<FineBalancerActionSample>
FineBalancerKpmStore::GetActionSamples() const
{
    return m_actionSamples;
}

std::vector<FineBalancerKpmSample>
FineBalancerKpmStore::GetKpmSamplesByStep(uint32_t step) const
{
    std::vector<FineBalancerKpmSample> samples;
    for (const auto& sample : m_kpmSamples)
    {
        if (sample.step == step)
        {
            samples.push_back(sample);
        }
    }
    return samples;
}

void
FineBalancerKpmStore::WriteJoinedCsv(const std::string& filename) const
{
    // Collect names in insertion order; split Param_* from KPMs
    std::vector<std::string> paramNames;
    std::vector<std::string> kpmNames;
    {
        std::set<std::string> seen;
        for (const auto& s : m_kpmSamples)
        {
            if (!seen.insert(s.kpmName).second)
            {
                continue;
            }
            if (s.kpmName.rfind("Param_", 0) == 0)
            {
                paramNames.push_back(s.kpmName);
            }
            else
            {
                kpmNames.push_back(s.kpmName);
            }
        }
    }

    std::map<uint32_t, std::map<std::string, double>> kpmByStep;
    for (const auto& s : m_kpmSamples)
    {
        kpmByStep[s.step][s.kpmName] = s.value;
    }

    std::ofstream output(filename);
    output << "step,time_ns,xapp,adjusted_param,adjusted_value";
    for (const auto& name : paramNames)
    {
        output << "," << name.substr(6); // strip "Param_"
    }
    for (const auto& name : kpmNames)
    {
        output << "," << name;
    }
    output << "\n";

    for (const auto& action : m_actionSamples)
    {
        output << action.step << "," << action.timestamp.GetNanoSeconds() << ","
               << action.xappName << "," << action.adjustedParam << ","
               << action.adjustedValue;
        const auto& kpms = kpmByStep[action.step];
        for (const auto& name : paramNames)
        {
            auto it = kpms.find(name);
            output << "," << (it != kpms.end() ? it->second : 0.0);
        }
        for (const auto& name : kpmNames)
        {
            auto it = kpms.find(name);
            output << "," << (it != kpms.end() ? it->second : 0.0);
        }
        output << "\n";
    }
}

} // namespace oran
} // namespace ns3
