#include <cassert>
#include <golly/Support/isl.h>

namespace islpp {

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

ostream &ostream::operator<<(const space &val) {
  apply<isl_printer_print_space>(val.get());
  return *this;
}

ostream &ostream::operator<<(const union_set &val) {
  apply<isl_printer_print_union_set>(val.get());
  return *this;
}

space::space(const space_config &names)
    : base{([&]() -> base {
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

union_set union_set::operator*(const union_set &rhs) const {
  return union_set{isl_union_set_intersect(get(), rhs.get())};
}

union_set union_set::operator-(const union_set &rhs) const {
  return union_set{isl_union_set_subtract(get(), rhs.get())};
}

boolean union_set::operator==(const union_set &rhs) const {
  return isl_union_set_is_equal(get(), rhs.get());
}

boolean union_set::is_empty() const { return isl_union_set_is_empty(get()); }

boolean union_set::operator<=(const union_set &rhs) const {
  return isl_union_set_is_subset(get(), rhs.get());
}

boolean union_set::operator>=(const union_set &rhs) const {
  return rhs <= *this;
}

boolean union_set::operator<(const union_set &rhs) const {
  return isl_union_set_is_strict_subset(get(), rhs.get());
}

boolean union_set::operator>(const union_set &rhs) const { return rhs > *this; }

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
} // namespace islpp