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

space::space(const space_config &names)
    : base{([&]() -> isl_space * {
        auto sp = isl_space_alloc(ctx(), names.params.size(), names.in.size(),
                                  names.out.size());

        auto set_dim_names = [&](isl_dim_type dim_type,
                                 const vector<string> &vals) {
          unsigned gen = 0;
          for (auto &elem : vals)
            sp = isl_space_set_dim_name(sp, dim_type, gen++, elem.data());
        };

        set_dim_names(isl_dim_type::isl_dim_param, names.params);
        set_dim_names(isl_dim_type::isl_dim_in, names.in);
        set_dim_names(isl_dim_type::isl_dim_out, names.params);
        return sp;
      })()} {}

// union_set
union_set::union_set(string_view str)
    : base{isl_union_set_read_from_str(ctx(), str.data())} {}

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

// union_map
union_map::union_map(string_view isl)
    : base{isl_union_map_read_from_str(ctx(), isl.data())} {}

// set
set::set(string_view isl) : base{isl_set_read_from_str(ctx(), isl.data())} {}

map::map(string_view isl) : base{isl_map_read_from_str(ctx(), isl.data())} {}
} // namespace islpp