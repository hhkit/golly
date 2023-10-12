#ifndef GOLLY_SUPPORT_BFS_H
#define GOLLY_SUPPORT_BFS_H

#include <concepts>
#include <queue>

namespace golly {
template <typename T>
void bfs(T &&entry, std::invocable<T> auto getChildren,
         std::invocable<T> auto visitor) {
  std::queue<T> q;
  q.push(entry);

  while (!q.empty()) {
    auto node = q.front();
    q.pop();
    visitor(node);

    for (auto &elem : getChildren(node))
      q.push(elem);
  }
}
} // namespace golly

#endif /* GOLLY_SUPPORT_BFS_H */
