# cocoa_roman_kl
Integrating KL into CoCoA (v4.04)

## Notes

- The corresponding `cosmolike_core` branch for the `generic_interface.cpp` parser fix is `nonlimber-dev`.
- A typical failure signature is:
  `[critical] read_table: failed to parse file external_modules/data/roman_kl/Roman_3x2pt_cov_Ncl20_Ntomo10 at data line 12401, column 9: token='1.630861e-316' (out of range)`
- `Roman_3x2pt_cov_Ncl20_Ntomo10` can contain extremely small subnormal entries such as `1.630861e-316`.
- The original `read_table` implementation in `cosmolike_core/cosmolike/generic_interface.cpp` used `std::stod`, which raised a range error on these finite underflowed values and stopped likelihood initialization.
- The fix is to parse table values with `std::strtod` and only treat range errors as fatal when the parsed result is non-finite. Finite underflowed values are accepted.
- If you see this error, switch `Cocoa/external_modules/code/cosmolike_core` to branch `nonlimber-dev` or apply the same `generic_interface.cpp` patch, then rebuild `projects/roman_kl/interface/cosmolike_roman_kl_interface.so`.
