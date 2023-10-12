#ifndef GOLLY_SUPPORT_ISL_H
#define GOLLY_SUPPORT_ISL_H
// a subset of isl wrapped up for me to use without killing myself

#include <initializer_list>
#include <isl/aff.h>
#include <isl/constraint.h>
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/point.h>
#include <isl/printer.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace islpp {
using std::initializer_list;
using std::string;
using std::string_view;
using std::vector;
using boolean = isl_bool;

namespace detail {
template <typename IslTy, auto... Fns> class wrap;
template <typename IslTy, auto Deleter> class wrap<IslTy, Deleter> {
public:
  using base = wrap;

  wrap() : wrap{nullptr} {}
  explicit wrap(IslTy *ptr) : ptr{ptr} {}
  wrap(wrap &&rhs) : ptr{rhs.ptr} { rhs.ptr = nullptr; }
  wrap &operator=(wrap &&rhs) {
    std::swap(ptr, rhs.ptr);
    return *this;
  }

  ~wrap() {
    if (ptr)
      Deleter(ptr);
  };

  IslTy *get() const { return ptr; };

  // yields ownership
  IslTy *yield() {
    auto store = ptr;
    ptr = nullptr;
    return store;
  };

  wrap &&move() { return static_cast<wrap &&>(*this); }

private:
  IslTy *ptr;
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

enum class dim {
  constant = isl_dim_cst,
  set = isl_dim_set,
  in = isl_dim_in,
  out = isl_dim_out,
  param = isl_dim_param,
  all = isl_dim_all,
};

struct space_config {
  vector<string> set;
  vector<string> params;
  vector<string> in;
  vector<string> out;
};

#define UN_PROP(TYPE, RET, OP, FUNC)                                           \
  inline RET OP(const TYPE &obj) { return RET{FUNC(obj.get())}; }

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

#define MAP_OPERATORS(MAP, SET)                                                \
  OPEN_UNOP(SET, MAP, wrap, isl_##MAP##_wrap);                                 \
  OPEN_UNOP(MAP, SET, unwrap, isl_##SET##_unwrap);                             \
  OPEN_BINOP(SET, SET, MAP, apply, isl_##SET##_apply);                         \
  OPEN_UNOP(SET, MAP, domain, isl_##MAP##_domain);                             \
  OPEN_UNOP(SET, MAP, range, isl_##MAP##_range);

#define NAME_DIM_OPERATION(TYPE)                                               \
  inline TYPE name(TYPE bs, dim on, string_view name) {                        \
    return TYPE{isl_##TYPE##_set_tuple_name(                                   \
        bs.yield(), static_cast<isl_dim_type>(on), name.data())};              \
  }

#define NAME_OPERATORS(MAP, SET)                                               \
  inline string name(SET bs) {                                                 \
    auto str = isl_##SET##_get_tuple_name(bs.get());                           \
    auto ret = string{str};                                                    \
    return ret;                                                                \
  }                                                                            \
  inline string name(MAP bs, dim on) {                                         \
    auto str =                                                                 \
        isl_##MAP##_get_tuple_name(bs.get(), static_cast<isl_dim_type>(on));   \
    auto ret = string{str};                                                    \
    return ret;                                                                \
  }

#define SET_OPERATORS(TYPE)                                                    \
  UN_PROP(TYPE, isl_bool, is_empty, isl_##TYPE##_is_empty)                     \
  CLOSED_BINOP(TYPE, operator*, isl_##TYPE##_intersect);                       \
  CLOSED_BINOP(TYPE, operator-, isl_##TYPE##_subtract);                        \
  CLOSED_BINOP(TYPE, operator+, isl_##TYPE##_union);                           \
  BIN_PROP(TYPE, operator==, isl_##TYPE##_is_equal);                           \
  BIN_PROP(TYPE, operator<=, isl_##TYPE##_is_subset);                          \
  REV_BINOP(TYPE, operator>=, operator<=);                                     \
  BIN_PROP(TYPE, operator<, isl_##TYPE##_is_strict_subset);                    \
  REV_BINOP(TYPE, operator>, operator<);                                       \
  CLOSED_UNOP(TYPE, coalesce, isl_##TYPE##_coalesce);                          \
  CLOSED_BINOP(TYPE, cross, isl_##TYPE##_product);

#define LEXICAL_OPERATORS(SET, MAP)                                            \
  OPEN_BINOP(MAP, SET, SET, operator<<, isl_##SET##_lex_lt_##SET);             \
  OPEN_BINOP(MAP, SET, SET, operator<<=, isl_##SET##_lex_le_##SET);            \
  OPEN_BINOP(MAP, SET, SET, operator>>, isl_##SET##_lex_gt_##SET);             \
  OPEN_BINOP(MAP, SET, SET, operator>>=, isl_##SET##_lex_ge_##SET);            \
  CLOSED_UNOP(SET, lexmin, isl_##SET##_lexmin);                                \
  CLOSED_UNOP(SET, lexmax, isl_##SET##_lexmax);

#define DIMS(TYPE)                                                             \
                                                                               \
  inline isl_size dims(const TYPE &s, dim on) {                                \
    return isl_##TYPE##_dim(s.get(), static_cast<isl_dim_type>(on));           \
  }

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

class multi_val : public detail::wrap<isl_multi_val, isl_multi_val_copy,
                                      isl_multi_val_free> {
public:
  using base::base;
  explicit multi_val(string v)
      : base{isl_multi_val_read_from_str(ctx(), v.data())} {}
};

inline multi_val name(multi_val m, string_view name) {
  return multi_val{isl_multi_val_set_tuple_name(
      m.yield(), isl_dim_type::isl_dim_cst, name.data())};
}

class union_map : public detail::wrap<isl_union_map, isl_union_map_copy,
                                      isl_union_map_free> {
public:
  using base::base;
  enum class consts { identity };
  explicit union_map(string_view isl = "{}");
};
MAP_OPERATORS(union_map, union_set);

CLOSED_UNOP(union_map, reverse, isl_union_map_reverse)
SET_OPERATORS(union_map)
OPEN_BINOP(union_map, union_map, val, fixed_power,
           isl_union_map_fixed_power_val);

OPEN_BINOP(union_map, union_set, union_set, universal,
           isl_union_map_from_domain_and_range);
OPEN_UNOP(union_map, union_set, identity, isl_union_set_identity);
OPEN_BINOP(union_map, union_map, union_set, domain_intersect,
           isl_union_map_intersect_domain);
OPEN_BINOP(union_map, union_map, union_set, range_intersect,
           isl_union_map_intersect_range);
UN_PROP(union_map, isl_bool, is_single_valued, isl_union_map_is_single_valued);
UN_PROP(union_map, isl_bool, is_injective, isl_union_map_is_injective);
UN_PROP(union_map, isl_bool, is_bijective, isl_union_map_is_bijective);
CLOSED_UNOP(union_map, zip, isl_union_map_zip);
CLOSED_BINOP(union_map, apply_range, isl_union_map_apply_range);
CLOSED_BINOP(union_map, apply_domain, isl_union_map_apply_domain);
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

  explicit operator union_set() const & {

    return union_set{isl_union_set_from_set(set{*this}.yield())};
  }
  explicit operator union_set() && {

    return union_set{isl_union_set_from_set(yield())};
  }
};

SET_OPERATORS(set)
CLOSED_BINOP(set, flat_cross, isl_set_flat_product);
CLOSED_UNOP(set, operator-, isl_set_neg);

DIMS(set)

inline set add_dims(set s, dim on, int count) {
  auto old_dim = dims(s, on);
  auto new_set =
      isl_set_add_dims(s.yield(), static_cast<isl_dim_type>(on), count);

  return set{new_set};
}

inline set add_dims(set s, dim on, initializer_list<string_view> vars) {
  auto old_dim = dims(s, on);
  auto new_set =
      isl_set_add_dims(s.yield(), static_cast<isl_dim_type>(on), vars.size());

  int index = old_dim;
  for (auto &elem : vars)
    new_set = isl_set_set_dim_name(new_set, static_cast<isl_dim_type>(on),
                                   index, elem.data());
  return set{new_set};
}

inline set project_out(set s, dim on, unsigned start, unsigned count) {
  return set{isl_set_project_out(s.yield(), static_cast<isl_dim_type>(on),
                                 start, count)};
}

inline set project_out(set s, dim on, initializer_list<string_view> vars) {
  auto res = s.get();
  for (auto &var : vars) {
    auto index = isl_set_find_dim_by_name(res, static_cast<isl_dim_type>(on),
                                          var.data());
    res = isl_set_project_out(res, static_cast<isl_dim_type>(on), index, 1);
  }
  return set{res};
}

inline set name(set s, string_view name) {
  return set{isl_set_set_tuple_name(s.yield(), name.data())};
}

class map : public detail::wrap<isl_map, isl_map_copy, isl_map_free> {
public:
  using base::base;

  map(string_view isl);

  explicit operator union_map() const & {

    return union_map{isl_union_map_from_map(map{*this}.yield())};
  }
  explicit operator union_map() && {

    return union_map{isl_union_map_from_map(yield())};
  }
};

DIMS(map)

inline map name(map m, dim on, string_view name) {
  return map{isl_map_set_tuple_name(m.yield(), static_cast<isl_dim_type>(on),
                                    name.data())};
}

inline map add_dims(map m, dim on, int count) {
  auto old_dim = dims(m, on);
  auto new_map =
      isl_map_add_dims(m.yield(), static_cast<isl_dim_type>(on), count);

  return map{new_map};
}
inline map project_onto(set s, dim on, int first, unsigned count) {
  return map{isl_set_project_onto_map(s.yield(), static_cast<isl_dim_type>(on),
                                      first, count)};
}

SET_OPERATORS(map)
CLOSED_UNOP(map, operator-, isl_map_neg);
OPEN_UNOP(map, set, identity, isl_set_identity);
MAP_OPERATORS(map, set);
CLOSED_UNOP(set, flatten, isl_set_flatten);
CLOSED_UNOP(map, flatten, isl_map_flatten);
CLOSED_BINOP(map, flat_cross, isl_map_flat_product);
CLOSED_BINOP(map, apply_range, isl_map_apply_range);
CLOSED_BINOP(map, apply_domain, isl_map_apply_domain);
CLOSED_UNOP(map, reverse, isl_map_reverse)
OPEN_BINOP(map, map, set, domain_intersect, isl_map_intersect_domain);
OPEN_BINOP(map, map, set, range_intersect, isl_map_intersect_range);

LEXICAL_OPERATORS(set, map);
LEXICAL_OPERATORS(map, map);

class point : public detail::wrap<isl_point, isl_point_copy, isl_point_free> {
public:
  using base::base;
};

#define SAMPLE(TYPE)                                                           \
  inline point sample(TYPE s) {                                                \
    return point{isl_##TYPE##_sample_point(s.yield())};                        \
  }

SAMPLE(set)
SAMPLE(union_set)

class basic_set : public detail::wrap<isl_basic_set, isl_basic_set_copy,
                                      isl_basic_set_free> {
public:
  using base::base;
  explicit operator set() const {
    return set{isl_set_from_basic_set(basic_set{*this}.yield())};
  }
  explicit operator union_set() const {
    return union_set{isl_union_set_from_basic_set(basic_set{*this}.yield())};
  }
};

class basic_map : public detail::wrap<isl_basic_map, isl_basic_map_copy,
                                      isl_basic_map_free> {
public:
  using base::base;

  explicit operator map() const {
    return map{isl_map_from_basic_map(basic_map{*this}.yield())};
  }
  explicit operator union_map() const {
    return union_map{isl_union_map_from_basic_map(basic_map{*this}.yield())};
  }
};

MAP_OPERATORS(basic_map, basic_set)
NAME_OPERATORS(basic_map, basic_set)
NAME_OPERATORS(map, set)

inline point sample(basic_set s) {
  return point{isl_basic_set_sample_point(s.yield())};
}
inline basic_map sample(union_map s) {
  return basic_map{isl_union_map_sample(s.yield())};
}
inline set drop_unused_params(set s) {
  return set{isl_set_drop_unused_params(s.get())};
}
inline basic_map clean(basic_map s) {
  auto dims = isl_basic_map_dim(s.get(), isl_dim_type::isl_dim_param);
  return basic_map{isl_basic_map_remove_dims(
      s.yield(), isl_dim_type::isl_dim_param, 0, dims)};
}

template <typename Fn> void for_each(const union_set &us, Fn &&fn) {
  isl_union_set_foreach_set(
      us.get(),
      +[](isl_set *st, void *user) -> isl_stat {
        using Ret = std::invoke_result_t<Fn, set>;
        if constexpr (std::is_same_v<Ret, isl_stat>)
          return (*static_cast<Fn *>(user))(set{st});
        else {
          (*static_cast<Fn *>(user))(set{st});
          return isl_stat_ok;
        }
      },
      &fn);
}

template <typename Fn> void for_each(const union_map &us, Fn &&fn) {
  isl_union_map_foreach_map(
      us.get(),
      +[](isl_map *st, void *user) -> isl_stat {
        using Ret = std::invoke_result_t<Fn, map>;
        if constexpr (std::is_same_v<Ret, isl_stat>)
          return (*static_cast<Fn *>(user))(map{st});
        else {
          (*static_cast<Fn *>(user))(map{st});
          return isl_stat_ok;
        }
      },
      &fn);
}

template <typename Fn> void scan(const union_set &us, Fn &&fn) {
  isl_union_set_foreach_point(
      us.get(),
      +[](isl_point *pt, void *user) -> isl_stat {
        using Ret = std::invoke_result_t<Fn, point>;
        if constexpr (std::is_same_v<Ret, isl_stat>)
          return (*static_cast<Fn *>(user))(point{pt});
        else {
          (*static_cast<Fn *>(user))(point{pt});
          return isl_stat_ok;
        }
      },
      &fn);
}
template <typename Fn> void scan(const set &us, Fn &&fn) {
  isl_set_foreach_point(
      us.get(),
      +[](isl_point *pt, void *user) -> isl_stat {
        using Ret = std::invoke_result_t<Fn, point>;
        if constexpr (std::is_same_v<Ret, isl_stat>)
          return (*static_cast<Fn *>(user))(point{pt});
        else {
          (*static_cast<Fn *>(user))(point{pt});
          return isl_stat_ok;
        }
      },
      &fn);
}

#define MAP_EXPRESSION(ID, MAP_TYPE)                                           \
  class ID : public detail::wrap<isl_##ID, isl_##ID##_copy, isl_##ID##_free> { \
  public:                                                                      \
    using base::base;                                                          \
    ID(string_view isl) : base{isl_##ID##_read_from_str(ctx(), isl.data())} {} \
    explicit operator MAP_TYPE() const {                                       \
      return MAP_TYPE(isl_##MAP_TYPE##_from_##ID(ID{*this}.yield()));          \
    }                                                                          \
  };                                                                           \
  CLOSED_BINOP(ID, operator+, isl_##ID##_add)                                  \
  CLOSED_BINOP(ID, operator-, isl_##ID##_sub)

#define SETMAP_EXPRESSION(ID, SET_TYPE, MAP_TYPE)                              \
  class ID : public detail::wrap<isl_##ID, isl_##ID##_copy, isl_##ID##_free> { \
  public:                                                                      \
    using base::base;                                                          \
    ID(string_view isl) : base{isl_##ID##_read_from_str(ctx(), isl.data())} {} \
    explicit operator SET_TYPE() const {                                       \
      return SET_TYPE(isl_##SET_TYPE##_from_##ID(ID{*this}.yield()));          \
    }                                                                          \
    explicit operator MAP_TYPE() const {                                       \
      return MAP_TYPE(isl_##MAP_TYPE##_from_##ID(ID{*this}.yield()));          \
    }                                                                          \
  };                                                                           \
  CLOSED_BINOP(ID, operator+, isl_##ID##_add)                                  \
  CLOSED_BINOP(ID, operator-, isl_##ID##_sub)

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

#define NE_OP(ID, TYPE)                                                        \
  OPEN_BINOP(TYPE, ID, ID, ne_##TYPE, isl_##ID##_ne_##TYPE)

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
MAP_EXPRESSION(aff, map);

class multi_aff : public detail::wrap<isl_multi_aff, isl_multi_aff_copy,
                                      isl_multi_aff_free> {
public:
  using base::base;
  multi_aff(string_view isl)
      : base{isl_multi_aff_read_from_str(ctx(), isl.data())} {}
  multi_aff(aff val) : base{isl_multi_aff_from_aff(val.yield())} {}
  explicit operator set() const {
    return set(isl_set_from_multi_aff(multi_aff{*this}.yield()));
  }
  explicit operator map() const {
    return map(isl_map_from_multi_aff(multi_aff{*this}.yield()));
  }
};
CLOSED_BINOP(multi_aff, operator+, isl_multi_aff_add)
CLOSED_BINOP(multi_aff, operator-, isl_multi_aff_sub)

class pw_aff
    : public detail::wrap<isl_pw_aff, isl_pw_aff_copy, isl_pw_aff_free> {
public:
  using base::base;
  explicit pw_aff(string_view isl)
      : base{isl_pw_aff_read_from_str(ctx(), isl.data())} {}
  explicit pw_aff(aff val) : base{isl_pw_aff_from_aff(val.yield())} {}

  explicit operator set() const {
    return set(isl_set_from_pw_aff(pw_aff{*this}.yield()));
  }
  explicit operator map() const {
    return map(isl_map_from_pw_aff(pw_aff{*this}.yield()));
  }
};
CLOSED_BINOP(pw_aff, operator+, isl_pw_aff_add)
CLOSED_BINOP(pw_aff, operator-, isl_pw_aff_sub)

SETMAP_EXPRESSION(pw_multi_aff, set, map);
SETMAP_EXPRESSION(multi_pw_aff, set, map);
MAP_EXPRESSION(union_pw_aff, union_map);
MAP_EXPRESSION(union_pw_multi_aff, union_map);
MAP_EXPRESSION(multi_union_pw_aff, union_map);

MINMAX_EXPR(pw_aff);
MINMAX_EXPR(multi_pw_aff);

DIMS(pw_aff)
DIMS(multi_aff)

inline multi_aff add_dims(multi_aff s, dim on, unsigned count) {
  return multi_aff{
      isl_multi_aff_add_dims(s.yield(), static_cast<isl_dim_type>(on), count)};
}

inline multi_aff drop_dims(multi_aff sp, dim d, int first, int n) {
  return multi_aff{isl_multi_aff_drop_dims(
      sp.yield(), static_cast<isl_dim_type>(d), first, n)};
}

inline pw_aff add_dims(pw_aff s, dim on, unsigned count) {
  return pw_aff{
      isl_pw_aff_add_dims(s.yield(), static_cast<isl_dim_type>(on), count)};
}

inline aff set_coeff(aff a, dim on, unsigned pos, int val) {
  return aff{isl_aff_set_coefficient_si(
      a.yield(), static_cast<isl_dim_type>(on), pos, val)};
}

OPEN_UNOP(set, pw_aff, domain, isl_pw_aff_domain)

// EXPRESSION(aff);
// EXPRESSION(multi_aff);
// EXPRESSION(pw_aff);
// EXPRESSION(pw_multi_aff);
// EXPRESSION(multi_pw_aff);
// EXPRESSION(union_pw_aff);
// EXPRESSION(union_pw_multi_aff);
// EXPRESSION(multi_union_pw_aff);

NAME_DIM_OPERATION(multi_aff)

COMPARABLE_EXPR(aff);
NE_OP(aff, set);
COMPARABLE_EXPR(pw_aff);
NE_OP(pw_aff, set);
COMPARE_OPS(pw_aff, map);
CLOSED_BINOP(pw_aff, operator/, isl_pw_aff_tdiv_q)
CLOSED_BINOP(pw_aff, operator%, isl_pw_aff_tdiv_r)

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

class space : public detail::wrap<isl_space, isl_space_copy, isl_space_free> {
public:
  using base::base;

  template <typename T> T zero() const;
  template <typename T> T constant(int val) const;
  template <typename T> T coeff(dim on, int pos, int val) const;
};

// represents a domain
class local_space : public detail::wrap<isl_local_space, isl_local_space_copy,
                                        isl_local_space_free> {
public:
  using base::base;

  local_space(space s) : base{isl_local_space_from_space(s.yield())} {}

  template <typename T> T zero() const;
  template <typename T> T constant(int val) const;
  template <typename T> T coeff(dim on, int pos, int val) const;
};

#define SPACE_OPS(TYPE) UN_PROP(TYPE, space, get_space, isl_##TYPE##_get_space);

#define DOMAIN_SPACE(TYPE)                                                     \
  UN_PROP(TYPE, space, domain, isl_##TYPE##_get_domain_space);

DIMS(space)
DIMS(local_space)

SPACE_OPS(basic_set)
SPACE_OPS(basic_map)
SPACE_OPS(set)
SPACE_OPS(map)
SPACE_OPS(union_set)
SPACE_OPS(union_map)
SPACE_OPS(aff)
SPACE_OPS(multi_aff)
SPACE_OPS(pw_aff)
SPACE_OPS(pw_multi_aff)
SPACE_OPS(multi_pw_aff)
SPACE_OPS(union_pw_aff)
SPACE_OPS(union_pw_multi_aff)
SPACE_OPS(multi_union_pw_aff)

DOMAIN_SPACE(aff)
DOMAIN_SPACE(multi_aff)
DOMAIN_SPACE(pw_aff)
DOMAIN_SPACE(pw_multi_aff)
DOMAIN_SPACE(multi_pw_aff)
DOMAIN_SPACE(multi_union_pw_aff)

template <typename T> local_space get_local_space(T &&val) {
  return local_space{get_space(std::forward<T>(val))};
}

template <typename To, typename From> inline To cast(From val) {
  static_assert(std::is_same_v<From, To>, "invalid cast");
  if constexpr (std::is_same_v<From, To>)
    return std::move(val);
}

#define CAST(FROM, TO)                                                         \
  template <> inline TO cast<TO, FROM>(FROM val) {                             \
    return TO{isl_##TO##_from_##FROM(val.yield())};                            \
  }

CAST(aff, multi_aff)
CAST(aff, pw_aff)

template <typename T> inline T space::zero() const {
  auto val = aff{isl_aff_zero_on_domain_space(space{*this}.yield())};
  return cast<T>(std::move(val));
}

template <typename T> inline T space::constant(int val) const {
  auto setted = aff{isl_aff_set_constant_si(zero<aff>().yield(), val)};
  return cast<T>(std::move(setted));
}

template <typename T> inline T space::coeff(dim on, int index, int val) const {
  auto setted = aff{isl_aff_set_coefficient_si(
      zero<aff>().yield(), static_cast<isl_dim_type>(on), index, val)};
  return cast<T>(std::move(setted));
}

template <typename T> inline T local_space::zero() const {
  auto val = aff{isl_aff_zero_on_domain(local_space{*this}.yield())};
  return cast<T>(std::move(val));
}

template <typename T> inline T local_space::constant(int val) const {
  auto setted = aff{isl_aff_set_constant_si(zero<aff>().yield(), val)};
  return cast<T>(std::move(setted));
}

template <typename T>
inline T local_space::coeff(dim on, int index, int val) const {
  auto setted = aff{isl_aff_set_coefficient_si(
      zero<aff>().yield(), static_cast<isl_dim_type>(on), index, val)};
  return cast<T>(std::move(setted));
}

inline space add_dims(space sp, dim on, int count) {
  return space{
      isl_space_add_dims(sp.yield(), static_cast<isl_dim_type>(on), count)};
}

inline space add_param(space sp, string_view param) {
  auto old_param_count = isl_space_dim(sp.get(), isl_dim_param);
  auto tmp = isl_space_add_dims(sp.yield(), isl_dim_type::isl_dim_param, 1);
  tmp =
      isl_space_set_dim_name(tmp, isl_dim_param, old_param_count, param.data());
  return space{tmp};
}

inline space drop_dims(space sp, dim d, int first, int n) {
  return islpp::space{
      isl_space_drop_dims(sp.yield(), static_cast<isl_dim_type>(d), first, n)};
}

// increments last val
inline multi_aff increment(multi_aff ma) {
  auto ret = ma;

  auto incre = domain(ma).constant<islpp::aff>(1);
  auto out = dims(ma, islpp::dim::out);
  auto last = islpp::aff{isl_multi_aff_get_aff(ma.get(), out - 1)};
  return islpp::multi_aff{
      isl_multi_aff_set_aff(ret.yield(), out - 1, (last + incre).yield())};
}

inline multi_aff project_up(multi_aff ma) {
  auto added = multi_aff{
      isl_multi_aff_add_dims(ma.yield(), isl_dim_type::isl_dim_in, 1)};
  auto ins = domain(added);
  auto wcoeff = ins.coeff<multi_aff>(dim::in, dims(added, dim::in) - 1, 1);

  return multi_aff{
      isl_multi_aff_flat_range_product(added.yield(), wcoeff.yield())};
}

inline multi_aff flat_range_product(std::span<aff> affs) {
  auto itr = affs.begin();
  islpp::multi_aff ret{*itr++};

  for (; itr != affs.end(); ++itr)
    ret = flat_range_product(ret, multi_aff{*itr});

  return ret;
}

inline multi_aff append_zero(multi_aff ma) {
  auto ins = domain(ma);
  auto zer = ins.zero<multi_aff>();

  return multi_aff{isl_multi_aff_flat_range_product(ma.yield(), zer.yield())};
}

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
  PRINT_DEF(basic_set)
  PRINT_DEF(basic_map)
  PRINT_DEF(space)
  PRINT_DEF(local_space)
  PRINT_DEF(val)
  PRINT_DEF(multi_val)
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
