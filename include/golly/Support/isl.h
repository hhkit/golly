#ifndef GOLLY_SUPPORT_ISL_H
#define GOLLY_SUPPORT_ISL_H
// a subset of isl wrapped up for me to use without killing myself

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/point.h>
#include <isl/printer.h>
#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace islpp {
using std::string;
using std::string_view;
using std::vector;
using boolean = isl_bool;

namespace detail {
template <typename IslTy, auto... Fns> class wrap;
template <typename IslTy, auto Deleter> class wrap<IslTy, Deleter> {
public:
  using base = wrap;

  explicit wrap(IslTy *ptr, bool owns = true) : ptr{ptr}, owns{owns} {}
  wrap(wrap &&rhs) : ptr{rhs.ptr}, owns{rhs.owns} { rhs.owns = false; }
  wrap &operator=(wrap &&rhs) {
    std::swap(ptr, rhs.ptr);
    std::swap(owns, rhs.owns);
    return *this;
  }

  ~wrap() {
    if (owns)
      Deleter(ptr);
  };

  IslTy *get() const { return ptr; };

  // yields ownership
  IslTy *yield() {
    owns = false;
    return ptr;
  };

private:
  IslTy *ptr;
  bool owns = true;
};

template <typename IslTy, auto Copier, auto Deleter>
class wrap<IslTy, Copier, Deleter> : public wrap<IslTy, Deleter> {
  using internal = wrap<IslTy, Deleter>;

public:
  using internal::get;
  using internal::internal;
  using internal::yield;
  using base = wrap;

  wrap(const wrap &rhs) : wrap{Copier(rhs.get())} {}
  wrap &operator=(const wrap &rhs) {
    auto newMe = internal{Copier(rhs.get())};
    internal::operator=(std::move(newMe));
    return *this;
  }
};

} // namespace detail

class isl_manager : public detail::wrap<isl_ctx, isl_ctx_free> {
public:
  isl_manager();
};

isl_ctx *ctx();

struct space_config {
  vector<string> set;
  vector<string> params;
  vector<string> in;
  vector<string> out;
};

#define UN_PROP(TYPE, OP, FUNC)                                                \
  inline auto OP(const TYPE &obj) { return FUNC(obj.get()); }

#define BIN_PROP(TYPE, OP, FUNC)                                               \
  inline auto OP(const TYPE &lhs, const TYPE &rhs) {                           \
    return (FUNC)(lhs.get(), rhs.get());                                       \
  }

#define OPEN_UNOP(TYPE_TO, TYPE_FROM, NAME, FUNC)                              \
  inline TYPE_TO NAME(TYPE_FROM obj) { return TYPE_TO{FUNC(obj.yield())}; }

#define OPEN_BINOP(TYPE_TO, TYPE_FROM1, TYPE_FROM2, NAME, FUNC)                \
  inline TYPE_TO NAME(TYPE_FROM1 lhs, TYPE_FROM2 rhs) {                        \
    return TYPE_TO{FUNC(lhs.yield(), rhs.yield())};                            \
  }

#define CLOSED_UNOP(TYPE, OP, FUNC) OPEN_UNOP(TYPE, TYPE, OP, FUNC)

#define CLOSED_BINOP(TYPE, OP, FUNC) OPEN_BINOP(TYPE, TYPE, TYPE, OP, FUNC)

#define REV_BINOP(TYPE, OP, OTHER)                                             \
  inline auto OP(const TYPE &lhs, const TYPE &rhs) { return OTHER(rhs, lhs); }

#define SET_OPERATORS(TYPE)                                                    \
  UN_PROP(TYPE, is_empty, isl_##TYPE##_is_empty)                               \
  CLOSED_BINOP(TYPE, operator*, isl_##TYPE##_intersect);                       \
  CLOSED_BINOP(TYPE, operator-, isl_##TYPE##_subtract);                        \
  CLOSED_BINOP(TYPE, operator+, isl_##TYPE##_union);                           \
  BIN_PROP(TYPE, operator==, isl_##TYPE##_is_equal);                           \
  BIN_PROP(TYPE, operator<=, isl_##TYPE##_is_subset);                          \
  REV_BINOP(TYPE, operator>=, operator<=);                                     \
  BIN_PROP(TYPE, operator<, isl_##TYPE##_is_strict_subset);                    \
  REV_BINOP(TYPE, operator>, operator<);                                       \
  CLOSED_UNOP(TYPE, coalesce, isl_##TYPE##_coalesce);

#define LEXICAL_OPERATORS(SET, MAP)                                            \
  OPEN_BINOP(MAP, SET, SET, operator<<, isl_##SET##_lex_lt_##SET);             \
  OPEN_BINOP(MAP, SET, SET, operator<<=, isl_##SET##_lex_le_##SET);            \
  OPEN_BINOP(MAP, SET, SET, operator>>, isl_##SET##_lex_gt_##SET);             \
  OPEN_BINOP(MAP, SET, SET, operator>>=, isl_##SET##_lex_ge_##SET);            \
  CLOSED_UNOP(SET, lexmin, isl_##SET##_lexmin);                                \
  CLOSED_UNOP(SET, lexmax, isl_##SET##_lexmax);

class union_set : public detail::wrap<isl_union_set, isl_union_set_copy,
                                      isl_union_set_free> {
public:
  using base::base;

  union_set() : union_set{"{}"} {}
  explicit union_set(string_view isl);
};
SET_OPERATORS(union_set)

class val : public detail::wrap<isl_val, isl_val_copy, isl_val_free> {
  enum class consts { zero, one, negone, nan, infty, neginfty };
  explicit val(consts);
  explicit val(long i);
  explicit val(unsigned long i);
};

class union_map : public detail::wrap<isl_union_map, isl_union_map_copy,
                                      isl_union_map_free> {
public:
  using base::base;
  enum class consts { identity };
  explicit union_map(string_view isl = "{}");
};

CLOSED_UNOP(union_map, reverse, isl_union_map_reverse)
SET_OPERATORS(union_map)
OPEN_BINOP(union_map, union_map, val, fixed_power,
           isl_union_map_fixed_power_val);
OPEN_UNOP(union_set, union_map, domain, isl_union_map_domain);
OPEN_UNOP(union_set, union_map, range, isl_union_map_range);
OPEN_BINOP(union_map, union_set, union_set, universal,
           isl_union_map_from_domain_and_range);
OPEN_UNOP(union_map, union_set, identity, isl_union_set_identity);
OPEN_BINOP(union_map, union_map, union_set, domain_subtract,
           isl_union_map_intersect_domain);
OPEN_BINOP(union_map, union_map, union_set, range_subtract,
           isl_union_map_intersect_range);
OPEN_BINOP(union_set, union_set, union_map, apply, isl_union_set_apply);

UN_PROP(union_map, is_single_valued, isl_union_map_is_single_valued);
UN_PROP(union_map, is_injective, isl_union_map_is_injective);
UN_PROP(union_map, is_bijective, isl_union_map_is_bijective);
OPEN_UNOP(union_set, union_map, wrap, isl_union_map_wrap);
OPEN_UNOP(union_map, union_set, unwrap, isl_union_set_unwrap);
CLOSED_BINOP(union_set, cross, isl_union_set_product);
CLOSED_BINOP(union_map, cross, isl_union_map_product);
CLOSED_UNOP(union_map, zip, isl_union_map_zip);
CLOSED_BINOP(union_map, domain_product, isl_union_map_domain_product);
CLOSED_BINOP(union_map, range_product, isl_union_map_range_product);
CLOSED_UNOP(union_map, domain_factor, isl_union_map_factor_domain);
CLOSED_UNOP(union_map, range_factor, isl_union_map_factor_range);
CLOSED_UNOP(union_map, domain_factor_domain,
            isl_union_map_domain_factor_domain);
CLOSED_UNOP(union_map, domain_factor_range, isl_union_map_range_factor_domain);
CLOSED_UNOP(union_map, range_factor_domain, isl_union_map_range_factor_domain);
CLOSED_UNOP(union_map, range_factor_range, isl_union_map_range_factor_range);
CLOSED_UNOP(union_map, domain_map, isl_union_map_domain_map);
CLOSED_UNOP(union_map, range_map, isl_union_map_range_map);
OPEN_UNOP(union_set, union_map, deltas, isl_union_map_deltas);
CLOSED_UNOP(union_map, deltas_map, isl_union_map_deltas_map);

LEXICAL_OPERATORS(union_set, union_map);
LEXICAL_OPERATORS(union_map, union_map);

// this class represents a unit set
// a set with only one tuple
class set : public detail::wrap<isl_set, isl_set_copy, isl_set_free> {
public:
  using base::base;

  set(string_view isl);
};
SET_OPERATORS(set)

class map : public detail::wrap<isl_map, isl_map_copy, isl_map_free> {
public:
  using base::base;

  map(string_view isl);
};
SET_OPERATORS(map)

LEXICAL_OPERATORS(set, map);
LEXICAL_OPERATORS(map, map);

class point : public detail::wrap<isl_point, isl_point_copy, isl_point_free> {
public:
  using base::base;
};

#define SAMPLE(TYPE)                                                           \
  inline point sample(TYPE s) {                                                \
    return point{isl_##TYPE##_sample_point(s.get())};                          \
  }

SAMPLE(set)
SAMPLE(union_set)

template <typename Fn> void for_each(const union_set &us, Fn &&fn) {
  isl_union_set_foreach_set(
      us.get(),
      +[](isl_set *st, void *user) { (*static_cast<Fn *>(user))(set{st}); },
      &fn);
}

template <typename Fn> void for_each(const union_map &us, Fn &&fn) {
  isl_union_map_foreach_map(
      us.get(),
      +[](isl_map *st, void *user) { (*static_cast<Fn *>(user))(map{st}); },
      &fn);
}

template <typename Fn> void scan(const union_set &us, Fn &&fn) {
  isl_union_set_foreach_point(
      us.get(),
      +[](isl_point *pt, void *user) { (*static_cast<Fn *>(user))(point{pt}); },
      &fn);
}
template <typename Fn> void scan(const set &us, Fn &&fn) {
  isl_set_foreach_point(
      us.get(),
      +[](isl_point *pt, void *user) { (*static_cast<Fn *>(user))(point{pt}); },
      &fn);
}

#define EXPRESSION(ID, MAP_TYPE)                                               \
  class ID : public detail::wrap<isl_##ID, isl_##ID##_copy, isl_##ID##_free> { \
  public:                                                                      \
    using base::base;                                                          \
    ID(string_view isl) : base{isl_##ID##_read_from_str(ctx(), isl.data())} {} \
    explicit operator MAP_TYPE() {                                             \
      return MAP_TYPE(isl_##MAP_TYPE##_from_##ID(ID{*this}.yield()));          \
    }                                                                          \
  };                                                                           \
  CLOSED_BINOP(ID, operator+, isl_##ID##_add)

#define PW_EXPR(ID) CLOSED_BINOP(ID, union_add, isl_##ID##_union_add)

#define MINMAX_EXPR(ID)                                                        \
  CLOSED_BINOP(ID, max, isl_##ID##_max)                                        \
  CLOSED_BINOP(ID, min, isl_##ID##_min)

#define COMPARE_OPS(ID, TYPE)                                                  \
  OPEN_BINOP(TYPE, ID, ID, lt_##TYPE, isl_##ID##_lt_##TYPE)                    \
  OPEN_BINOP(TYPE, ID, ID, le_##TYPE, isl_##ID##_le_##TYPE)                    \
  OPEN_BINOP(TYPE, ID, ID, gt_##TYPE, isl_##ID##_gt_##TYPE)                    \
  OPEN_BINOP(TYPE, ID, ID, ge_##TYPE, isl_##ID##_ge_##TYPE)                    \
  OPEN_BINOP(TYPE, ID, ID, eq_##TYPE, isl_##ID##_eq_##TYPE)

#define COMPARABLE_EXPR(ID)                                                    \
  COMPARE_OPS(ID, set)                                                         \
  CLOSED_BINOP(ID, operator*, isl_##ID##_mul)

#define TUPLE_EXPR(ID) CLOSED_BINOP(ID, product, isl_##ID##_product)
#define MULTI_EXPR(ID)                                                         \
  CLOSED_BINOP(ID, flat_range_product, isl_##ID##_flat_range_product)          \
  CLOSED_BINOP(ID, range_product, isl_##ID##_range_product)

#define PULLBACK(FROM, TO)                                                     \
  OPEN_BINOP(FROM, FROM, TO, pullback, isl_##FROM##_pullback_##TO)

// okay who designed this?
EXPRESSION(aff, map);
EXPRESSION(multi_aff, map);
EXPRESSION(pw_aff, map);
EXPRESSION(pw_multi_aff, map);
EXPRESSION(multi_pw_aff, map);
EXPRESSION(union_pw_aff, union_map);
EXPRESSION(union_pw_multi_aff, union_map);
EXPRESSION(multi_union_pw_aff, union_map);

MINMAX_EXPR(pw_aff);
MINMAX_EXPR(multi_pw_aff);

OPEN_UNOP(set, pw_aff, domain, isl_pw_aff_domain)

// EXPRESSION(aff);
// EXPRESSION(multi_aff);
// EXPRESSION(pw_aff);
// EXPRESSION(pw_multi_aff);
// EXPRESSION(multi_pw_aff);
// EXPRESSION(union_pw_aff);
// EXPRESSION(union_pw_multi_aff);
// EXPRESSION(multi_union_pw_aff);

COMPARABLE_EXPR(aff);
COMPARABLE_EXPR(pw_aff);
COMPARE_OPS(pw_aff, map);

PW_EXPR(pw_aff);
PW_EXPR(pw_multi_aff);
PW_EXPR(multi_pw_aff);
PW_EXPR(union_pw_multi_aff);
PW_EXPR(union_pw_aff);
PW_EXPR(multi_union_pw_aff);

MULTI_EXPR(multi_aff);
MULTI_EXPR(pw_multi_aff);
MULTI_EXPR(multi_pw_aff);
MULTI_EXPR(union_pw_multi_aff);
MULTI_EXPR(multi_union_pw_aff);

TUPLE_EXPR(multi_aff);
TUPLE_EXPR(pw_multi_aff);
TUPLE_EXPR(multi_pw_aff);

// pullbacks
PULLBACK(aff, aff);
PULLBACK(aff, multi_aff);

PULLBACK(pw_aff, multi_aff);
PULLBACK(pw_aff, pw_multi_aff);
PULLBACK(pw_aff, multi_pw_aff);

PULLBACK(multi_aff, multi_aff);

PULLBACK(pw_multi_aff, multi_aff);
PULLBACK(pw_multi_aff, pw_multi_aff);

PULLBACK(multi_pw_aff, multi_aff);
PULLBACK(multi_pw_aff, pw_multi_aff);
PULLBACK(multi_pw_aff, multi_pw_aff);

PULLBACK(union_pw_aff, union_pw_multi_aff);

PULLBACK(union_pw_multi_aff, union_pw_multi_aff);
PULLBACK(multi_union_pw_aff, union_pw_multi_aff);

#define PRINT_DEF(TYPE)                                                        \
  ostream &operator<<(const TYPE &v) {                                         \
    apply<isl_printer_print_##TYPE>(v.get());                                  \
    return *this;                                                              \
  };

class ostream : public detail::wrap<isl_printer, isl_printer_free> {
public:
  PRINT_DEF(union_set)
  PRINT_DEF(union_map)
  PRINT_DEF(set)
  PRINT_DEF(map)
  PRINT_DEF(val)
  PRINT_DEF(point)
  PRINT_DEF(aff)
  PRINT_DEF(pw_aff)
  PRINT_DEF(multi_aff)
  PRINT_DEF(pw_multi_aff)
  PRINT_DEF(multi_pw_aff)
  PRINT_DEF(union_pw_aff)
  PRINT_DEF(union_pw_multi_aff)
  PRINT_DEF(multi_union_pw_aff)

protected:
  using base::base;
  template <auto Fn, typename... Args> void apply(Args &&...args) {
    auto next = Fn(yield(), args...);
    base::operator=(base{next});
  }
};

#undef UN_PROP
#undef BIN_PROP
#undef OPEN_UNOP
#undef OPEN_BINOP
#undef CLOSED_UNOP
#undef CLOSED_BINOP
#undef REV_BINOP
#undef SET_OPERATORS
#undef LEXICAL_OPERATORS
#undef SAMPLE
#undef PRINT_DEF

class osstream : public ostream {
public:
  osstream();
  ~osstream() noexcept;

  void flush();
  string str();
};

} // namespace islpp

#endif /* GOLLY_SUPPORT_ISL_H */
