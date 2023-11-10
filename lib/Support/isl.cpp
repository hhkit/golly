#include <cassert>
#include <golly/Support/isl.h>
#include <memory>
namespace islpp {
using std::unique_ptr;

isl_manager &getManager() {
  static unique_ptr<isl_manager> singleton{};
  if (!singleton)
    singleton = std::make_unique<isl_manager>();

  return *singleton;
}

isl_ctx *ctx() { return getManager().get(); }

std::optional<std::string> getLastError() {
  if (isl_ctx_last_error(ctx()) != isl_error::isl_error_none) {
    auto last_err = isl_ctx_last_error_msg(ctx());
    isl_ctx_reset_error(ctx());
    auto ret = std::string(last_err);
    // free((void *)last_err);
    return last_err;
  }
  return std::nullopt;
}
// isl_manager
isl_manager::isl_manager() : base{isl_ctx_alloc()} {}

// printer

osstream::osstream() : ostream{isl_printer_to_str(ctx())} {
  apply<isl_printer_set_output_format>(ISL_FORMAT_ISL);
};

osstream::~osstream(){};

void osstream::flush() { apply<isl_printer_flush>(); }

string osstream::str() {
  auto ptr = isl_printer_get_str(get());
  flush();
  string retval{ptr};
  free(ptr);
  return retval;
}

// val
val::val(consts val)
    : base{([](consts val) -> isl_val * {
        switch (val) {
        case consts::zero:
          return isl_val_zero(ctx());
        case consts::one:
          return isl_val_one(ctx());
        case consts::negone:
          return isl_val_negone(ctx());
        case consts::nan:
          return isl_val_nan(ctx());
        case consts::infty:
          return isl_val_infty(ctx());
        case consts::neginfty:
          return isl_val_neginfty(ctx());
        default:
          return nullptr;
        }
      })(val)} {}
val::val(long i) : base{isl_val_int_from_si(ctx(), i)} {}
val::val(unsigned long i) : base{isl_val_int_from_ui(ctx(), i)} {}

union_set::union_set(string_view str)
    : base{isl_union_set_read_from_str(ctx(), str.data())} {}
union_map::union_map(string_view isl)
    : base{isl_union_map_read_from_str(ctx(), isl.data())} {}

set::set(string_view isl) : base{isl_set_read_from_str(ctx(), isl.data())} {}
map::map(string_view isl) : base{isl_map_read_from_str(ctx(), isl.data())} {}
} // namespace islpp