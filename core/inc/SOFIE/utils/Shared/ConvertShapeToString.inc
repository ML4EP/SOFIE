inline std::string ConvertShapeToString(const std::vector<std::size_t> &shape)
{
   std::stringstream out;
   out << "{ ";
   for (std::size_t i = 0; i < shape.size(); i++) {
      out << shape[i];
      if (i < shape.size() - 1)
         out << " , ";
   }
   out << " }";
   return out.str();
}
