find_library(ARMADILLO armadillo)
find_library(MLPACK mlpack)
find_path(ENSMALLEN_INCLUDE_DIR ensmallen.hpp)
set(mlpack_libraries)
set(oran_has_mlpack FALSE)
if(NOT(${ARMADILLO} STREQUAL ARMADILLO-NOTFOUND)
   AND NOT(${MLPACK} STREQUAL MLPACK-NOTFOUND)
   AND ENSMALLEN_INCLUDE_DIR)
    set(mlpack_libraries ${ARMADILLO} ${MLPACK})
    set(oran_has_mlpack TRUE)
endif()

set(oran_mlpack_sources)
set(oran_mlpack_headers)
if(oran_has_mlpack)
    set(oran_mlpack_sources model/xAppHandoverMlpackKmeans.cc)
    set(oran_mlpack_headers model/xAppHandoverMlpackKmeans.h)
endif()

build_lib(
  LIBNAME oran
  SOURCE_FILES
    model/PubSubInfra.cc
    model/E2AP.cc
    #model/E2SM-KPM.cc  -- these are part of E2AP. separated only for readability.
    #model/E2SM-RC.cc   /
    model/E2SM-KPM-measurements.cc
    model/ORAN-message-types.cc
    model/finebalancer-kpm-store.cc
    model/xAppHandover.cc
    model/xAppHandoverMaxRsrq.cc
    ${oran_mlpack_sources}
  HEADER_FILES
    model/json.hpp
    model/PubSubInfra.h
    model/E2AP.h
    model/xApp.h
    model/E2SM-KPM-measurements.h
    model/E2SM-RC-control-types.h
    model/E2SM-RC-indication-types.h
    model/ORAN-message-types.h
    model/finebalancer-kpm-store.h
    model/xAppHandover.h
    model/xAppHandoverMaxRsrq.h
    ${oran_mlpack_headers}
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${mlpack_libraries}
)
