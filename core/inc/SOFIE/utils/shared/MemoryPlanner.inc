struct TensorLifeInfo {
   int begin;
   int end;
   std::size_t size;
};

struct MemoryResult {
   std::size_t total_bytes = 0;
   std::vector<std::size_t> offsets;
   std::size_t total_free_bytes = 0;
   std::size_t largest_free_block = 0;
};

namespace memory_detail {
struct FreeBlock {
   std::size_t offset;
   std::size_t size;
   bool operator<(const FreeBlock &other) const { return offset < other.offset; }
};
struct MemoryEvent {
   int t;
   int type;
   int idx;
   bool operator<(const MemoryEvent &o) const
   {
      if (t != o.t)
         return t < o.t;
      return type < o.type;
   }
};
}

inline MemoryResult OrganizeMemory(const std::vector<TensorLifeInfo> &tensorsInfo)
{
   using memory_detail::FreeBlock;
   using memory_detail::MemoryEvent;
   for (const auto &t : tensorsInfo) {
      if (!(t.end > t.begin)) {
         throw std::runtime_error("Each tensor must have end > begin.");
      }
   }

   std::vector<MemoryEvent> events;
   events.reserve(tensorsInfo.size() * 2);
   for (int i = 0; i < (int)tensorsInfo.size(); ++i) {
      events.push_back({tensorsInfo[i].end, 0, i});
      events.push_back({tensorsInfo[i].begin, 1, i});
   }
   std::sort(events.begin(), events.end());

   std::vector<std::size_t> tensorsOffset(tensorsInfo.size());
   std::set<FreeBlock> free_list;
   std::unordered_map<int, std::size_t> live_size;
   std::unordered_map<int, std::size_t> live_offset;
   std::size_t total_bytes = 0;
   std::size_t total_free_bytes = 0;
   std::size_t largest_free_block = 0;

   auto allocate_best_fit = [&](std::size_t need) -> std::size_t {
      auto best = free_list.end();
      for (auto it = free_list.begin(); it != free_list.end(); ++it) {
         if (it->size >= need) {
            if (best == free_list.end() || it->size < best->size)
               best = it;
         }
      }
      if (best != free_list.end()) {
         std::size_t off = best->offset;
         if (best->size == need) {
            free_list.erase(best);
         } else {
            FreeBlock updated{best->offset + need, best->size - need};
            free_list.erase(best);
            free_list.insert(updated);
         }
         return off;
      }
      std::size_t off = total_bytes;
      total_bytes += need;
      return off;
   };

   auto try_coalesce = [&](std::set<FreeBlock>::iterator it) {
      if (it != free_list.begin()) {
         auto prev = std::prev(it);
         if (prev->offset + prev->size == it->offset) {
            FreeBlock merged{prev->offset, prev->size + it->size};
            free_list.erase(prev);
            it = free_list.erase(it);
            it = free_list.insert(merged).first;
         }
      }
      auto next = std::next(it);
      if (next != free_list.end() && it->offset + it->size == next->offset) {
         FreeBlock merged{it->offset, it->size + next->size};
         free_list.erase(next);
         it = free_list.erase(it);
         free_list.insert(merged);
      }
   };

   for (const auto &e : events) {
      if (e.type == 0) {
         auto it_sz = live_size.find(e.idx);
         auto it_off = live_offset.find(e.idx);
         if (it_sz != live_size.end() && it_off != live_offset.end()) {
            FreeBlock fb{it_off->second, it_sz->second};
            auto it = free_list.insert(fb).first;
            try_coalesce(it);
            live_size.erase(it_sz);
            live_offset.erase(it_off);
         }
      } else {
         auto &t = tensorsInfo[e.idx];
         std::size_t footprint_before = total_bytes;
         std::size_t off = allocate_best_fit(t.size);
         tensorsOffset[e.idx] = off;
         live_size[e.idx] = t.size;
         live_offset[e.idx] = off;
         if (total_bytes > footprint_before) {
            total_free_bytes = 0;
            largest_free_block = 0;
            for (const auto &fb : free_list) {
               total_free_bytes += fb.size;
               largest_free_block = std::max(largest_free_block, fb.size);
            }
         }
      }
   }

   return MemoryResult{total_bytes, std::move(tensorsOffset), total_free_bytes, largest_free_block};
}
