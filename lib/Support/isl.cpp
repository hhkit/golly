#include <cassert>
#include <golly/Support/isl.h>
#include <isl/ctx.h>

namespace islpp {

#define PIMPL_DEF(KLASS)                                                       \
  KLASS::KLASS() : pimpl_{std::make_unique<impl>()} {}                         \
  KLASS::~KLASS() noexcept = default;                                          \
  KLASS::KLASS(KLASS &&) noexcept = default;                                   \
  KLASS &KLASS::operator=(KLASS &&) noexcept = default;                        \
  // KLASS::KLASS(const KLASS &rhs)                                               \
      : pimpl_{std::make_unique<impl>(*rhs.pimpl_)} {}                         \
  KLASS &KLASS::operator=(const KLASS &rhs) {                                  \
    pimpl_ = std::make_unique<impl>(*rhs.pimpl_);                              \
    return *this;                                                              \
  }

isl_manager &getManager() {
  static unique_ptr<isl_manager> singleton{};
  if (!singleton)
    singleton = std::make_unique<isl_manager>();

  return *singleton;
}

isl_ctx *ctx() { return getManager().ctx(); }

class isl_manager::impl {
public:
  unique_ptr<isl_ctx, decltype(&isl_ctx_free)> context;

  impl() : context{isl_ctx_alloc(), &isl_ctx_free} {}
};

PIMPL_DEF(isl_manager);

isl_ctx *isl_manager::ctx() { return pimpl_->context.get(); }

// val::val(consts c) {}

// PIMPL_DEF(val);
} // namespace islpp