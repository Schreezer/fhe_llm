function(target_set_warnings TARGET)
  set(MSVC_WARNINGS
      /W4 # Baseline reasonable warnings
      /permissive- # standards conformance mode for MSVC compiler
  )

  set(CLANG_WARNINGS
      -Wall
      -Wextra # reasonable and standard
      -Wshadow # warn the user if a variable declaration shadows one from a parent context
  )

  set(GCC_WARNINGS
      ${CLANG_WARNINGS}
  )

  if(MSVC)
    target_compile_options(${TARGET} PRIVATE ${MSVC_WARNINGS})
  elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    target_compile_options(${TARGET} PRIVATE ${CLANG_WARNINGS})
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${TARGET} PRIVATE ${GCC_WARNINGS})
  else()
    message(AUTHOR_WARNING "No compiler warnings set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
  endif()
endfunction()

function(target_set_warnings_as_errors TARGET)
  if(MSVC)
    target_compile_options(${TARGET} PRIVATE /WX)
  else()
    target_compile_options(${TARGET} PRIVATE -Werror)
  endif()
endfunction()
