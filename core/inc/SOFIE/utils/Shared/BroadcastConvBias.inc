template <typename T>
T *BroadcastConvBias(const T *data, const std::size_t channel, const std::vector<std::size_t> &targetShape)
{
   std::size_t size = targetShape.size();
   if (targetShape[1] != channel) {
      std::stringstream ss;
      ss << "SOFIE - Error broadcasting Conv Bias of shape {";
      ss << std::to_string(channel);
      ss << "} to ";
      ss << ConvertShapeToString(targetShape);
      throw std::runtime_error(ss.str());
   }

   std::size_t targetLength = ConvertShapeToLength(targetShape);
   T *newData = new T[targetLength];

   if (targetLength == channel) {
      std::copy(data, data + channel, newData);
      return newData;
   }

   std::size_t cStride = 1;
   for (std::size_t i = 2; i < size; i++)
      cStride *= targetShape[i];
   for (std::size_t i = 0; i < channel; i++) {
      std::fill(newData + i * cStride, newData + (i + 1) * cStride, data[i]);
   }
   std::size_t batch = targetShape[0];
   std::size_t bStride = channel * cStride;
   for (std::size_t i = 1; i < batch; i++) {
      std::copy(newData, newData + bStride, newData + i * bStride);
   }
   return newData;
}
