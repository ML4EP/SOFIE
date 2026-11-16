inline std::size_t ConvertShapeToLength(const std::vector<std::size_t> &shape)
{
   std::size_t length = 1;
   for (auto &dim : shape)
      length *= dim;
   return length;
}
