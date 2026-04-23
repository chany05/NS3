#ifndef FINEBALANCER_KPM_STORE_H
#define FINEBALANCER_KPM_STORE_H

#include "ns3/nstime.h"
#include "ns3/object.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ns3
{
namespace oran
{

struct FineBalancerKpmSample
{
    uint32_t step{0};
    Time timestamp;
    uint16_t cellId{0};
    uint16_t rnti{0};
    uint64_t imsi{0};
    std::string kpmName;
    double value{0.0};
};

struct FineBalancerActionSample
{
    uint32_t step{0};
    Time timestamp;
    std::string xappName;
    std::string adjustedParam;
    double adjustedValue{0.0};
    uint16_t cellId{0};
    uint16_t rnti{0};
};

class FineBalancerKpmStore : public Object
{
  public:
    static TypeId GetTypeId();
    static Ptr<FineBalancerKpmStore> Get();

    void Clear();
    void PutKpm(uint32_t step,
                uint16_t cellId,
                uint16_t rnti,
                uint64_t imsi,
                const std::string& kpmName,
                double value);
    void PutAction(uint32_t step,
                   const std::string& xappName,
                   const std::string& adjustedParam,
                   double adjustedValue,
                   uint16_t cellId = 0,
                   uint16_t rnti = 0);

    std::vector<FineBalancerKpmSample> GetKpmSamples() const;
    std::vector<FineBalancerActionSample> GetActionSamples() const;
    std::vector<FineBalancerKpmSample> GetKpmSamplesByStep(uint32_t step) const;

    void WriteJoinedCsv(const std::string& filename) const;

  private:
    std::vector<FineBalancerKpmSample> m_kpmSamples;
    std::vector<FineBalancerActionSample> m_actionSamples;
};

} // namespace oran
} // namespace ns3

#endif /* FINEBALANCER_KPM_STORE_H */
