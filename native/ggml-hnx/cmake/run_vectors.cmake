# Generate cross-check vectors with Python, then feed them to the C
# selftest. A separate script because ctest runs one command per test and
# this needs two, and because it lets the "Python cannot import
# hypernix" case be a skip rather than a failure -- a machine with only
# a C compiler should still get the self-contained checks.
execute_process(
    COMMAND ${PYTHON} ${GEN} ${WORKDIR}/vectors.bin
    RESULT_VARIABLE gen_result
    OUTPUT_VARIABLE gen_output
    ERROR_VARIABLE gen_error
)
if(NOT gen_result EQUAL 0)
    message(STATUS "${gen_output}${gen_error}")
    message(STATUS "could not generate vectors; skipping the cross-check")
    # ctest reads this as a skip when the test is configured with
    # SKIP_RETURN_CODE, and as a pass otherwise. Either way it does not
    # fail a build for a missing optional dependency.
    return()
endif()

execute_process(
    COMMAND ${SELFTEST} ${WORKDIR}/vectors.bin
    RESULT_VARIABLE test_result
)
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "the C decoder disagrees with hypernix.quant.subbit")
endif()
