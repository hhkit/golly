#ifndef GOLLY_SUPPORT_ISL_H
#define GOLLY_SUPPORT_ISL_H
#include <isl/ctx.h>
#include <memory>
namespace islpp {
using std::unique_ptr;

#define PIMPL_DECL(KLASS)                                                      \
private:                                                                       \
  class impl;                                                                  \
  unique_ptr<impl> pimpl_;                                                     \
                                                                               \
public:                                                                        \
  KLASS();                                                                     \
  ~KLASS() noexcept;                                                           \
  KLASS(KLASS &&) noexcept;                                                    \
  KLASS &operator=(KLASS &&);                                                  \
  // KLASS(const KLASS &);                                                        \
  // KLASS &operator=(const KLASS &);                                             \

class isl_manager {
  PIMPL_DECL(isl_manager);

public:
  isl_ctx *ctx();
};

isl_ctx *ctx();

class val {
public:
  enum class consts { zero, one, negone, nan, infty, neginfty };
  val(consts);

  PIMPL_DECL(val);
};
} // namespace islpp

#endif /* GOLLY_SUPPORT_ISL_H */
