#ifndef GOLLY_ADT_PAIRSET_H
#define GOLLY_ADT_PAIRSET_H

#include <unordered_set>

namespace golly {
template <typename T1, typename T2> struct PairHash {
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto lhash = std::hash<T1>{}(p.first);
    auto rhash = std::hash<T2>{}(p.second);
    auto s1 = (lhash + 0x9e3779b9);
    return (rhash + 0x9e3779b9) + (s1 << 6) + (s1 >> 2);
  }
};

template <typename T1, typename T2>
using PairSet = std::unordered_set<std::pair<T1, T2>, PairHash<T1, T2>>;
} // namespace golly

#endif /* GOLLY_ADT_PAIRSET_H */
