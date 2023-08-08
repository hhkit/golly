#ifndef GOLLY_SUPPORT_TYPE_LIST_H
#define GOLLY_SUPPORT_TYPE_LIST_H
#include <cstddef>
#include <utility>

namespace golly {
namespace detail {
template <typename... Ts> struct type_list;

template <typename Head, typename... Rest> struct type_list<Head, Rest...> {

  template <typename T> static constexpr size_t index_of() {
    if constexpr (std::is_same_v<T, Head>)
      return 0;
    else
      return 1 + type_list<Rest...>::template index_of<T>();
  }

  static constexpr auto length = 1 + sizeof...(Rest);
};
}; // namespace detail
} // namespace golly

#endif /* GOLLY_SUPPORT_TYPE_LIST_H */
