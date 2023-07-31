#ifndef GOLLY_SUPPORT_ISL_H
#define GOLLY_SUPPORT_ISL_H
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/printer.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_set.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace islpp {
using std::string;
using std::string_view;
using std::unique_ptr;
using std::vector;
using boolean = isl_bool;

class space;
class union_set;
class val;
class map;

namespace detail {
template <typename IslTy, auto... Fns> class wrap;

template <typename IslTy, auto Deleter>
class wrap<IslTy, Deleter> : protected unique_ptr<IslTy, decltype(Deleter)> {
protected:
  using internal = unique_ptr<IslTy, decltype(Deleter)>;

public:
  using base = wrap;

  wrap(IslTy *ptr) : internal{ptr, Deleter} {}
  using internal::get;
};

template <typename IslTy, auto Copier, auto Deleter>
class wrap<IslTy, Copier, Deleter> : unique_ptr<IslTy, decltype(Deleter)> {
  using internal = unique_ptr<IslTy, decltype(Deleter)>;

public:
  using base = wrap;

  wrap(IslTy *ptr) : internal{ptr, Deleter} {}
  wrap(const wrap &rhs) : wrap{Copier(rhs.get())} {}
  wrap &operator=(const wrap &rhs) {
    auto newMe = internal{Copier(rhs.get())};
    internal::operator=(std::move(newMe));
    return *this;
  }

  using internal::get;
};

} // namespace detail
class isl_manager : public detail::wrap<isl_ctx, isl_ctx_free> {
public:
  isl_manager();
};

isl_ctx *ctx();

class ostream : public detail::wrap<isl_printer, isl_printer_free> {
public:
  ostream &operator<<(const space &space);
  ostream &operator<<(const union_set &val);

protected:
  using base::base;
  template <auto Fn, typename... Args> void apply(Args &&...args) {
    auto next = Fn(internal::release(), args...);
    internal::operator=(internal{next, get_deleter()});
  }
};

class osstream : public ostream {
public:
  osstream();
  ~osstream() noexcept;

  void flush();
  string str();
};

struct space_config {
  vector<string> set;
  vector<string> params;
  vector<string> in;
  vector<string> out;
};

class space : public detail::wrap<isl_space, isl_space_free> {
public:
  explicit space(const space_config &cfg);
};

class union_set : public detail::wrap<isl_union_set, isl_union_set_free> {
public:
  using base::base;

  explicit union_set(string_view isl = "{}");

  union_set operator*(const union_set &rhs) const; // set intersection
  union_set operator-(const union_set &rhs) const; // set difference
  boolean operator==(const union_set &rhs) const;  // equality comparison
  boolean is_empty() const;
  boolean operator<=(const union_set &rhs) const; // is subset of
  boolean operator>=(const union_set &rhs) const;
  boolean operator<(const union_set &rhs) const; // is strict subset of
  boolean operator>(const union_set &rhs) const;
};

class map : public detail::wrap<isl_map, isl_map_free> {
public:
  explicit map(string_view isl = "{}");
};

class val : public detail::wrap<isl_val, isl_val_copy, isl_val_free> {
  enum class consts { zero, one, negone, nan, infty, neginfty };
  explicit val(consts);
  explicit val(long i);
  explicit val(unsigned long i);
};
} // namespace islpp

#endif /* GOLLY_SUPPORT_ISL_H */
