template <typename T>
void BroadcastTensor(const T *data, std::size_t curLength, const std::vector<std::size_t> &shape,
                     const std::vector<std::size_t> &targetShape, T *broadcastedData)
{
   std::size_t size = shape.size();
   if (size > 1 && shape.front() == targetShape.front() && shape.back() == 1) {
      std::size_t bsize = targetShape.back();
      for (int k = int(size) - 2; k >= 0; k--) {
         if (shape[k] != 1)
            break;
         bsize *= targetShape[k];
      }
      for (std::size_t i = 0; i < curLength; i++) {
         std::fill(broadcastedData + i * bsize, broadcastedData + (i + 1) * bsize, data[i]);
      }
      return;
   }

   std::copy(data, data + curLength, broadcastedData);
   std::size_t arrayNum = 1;
   std::vector<T> newData(ConvertShapeToLength(targetShape));

   for (std::size_t idx = 0; idx < size; idx++) {
      std::size_t dim = shape[idx];
      std::size_t targetDim = targetShape[idx];
      if (dim == 1 && targetDim > 1) {
         std::size_t newLength = curLength * targetDim;
         std::size_t arrayLength = curLength / arrayNum;
         if (arrayLength > 1) {
            for (std::size_t arrayIdx = 0; arrayIdx < arrayNum; arrayIdx++) {
               for (std::size_t targetIdx = 0; targetIdx < targetDim; targetIdx++) {
                  std::size_t offset = arrayIdx * arrayLength * targetDim + targetIdx * arrayLength;
                  std::copy(broadcastedData + arrayIdx * arrayLength,
                            broadcastedData + (arrayIdx + 1) * arrayLength,
                            newData.begin() + offset);
               }
            }
         } else {
            for (std::size_t arrayIdx = 0; arrayIdx < arrayNum; arrayIdx++) {
               std::fill(newData.begin() + arrayIdx * targetDim,
                         newData.begin() + (arrayIdx + 1) * targetDim, broadcastedData[arrayIdx]);
            }
         }
         curLength = newLength;
         std::copy(newData.begin(), newData.begin() + newLength, broadcastedData);
      }
      arrayNum *= targetDim;
   }
}

template <typename T>
T *CreateBroadcastTensor(const T *data, const std::vector<std::size_t> &shape,
                         const std::vector<std::size_t> &targetShape, std::size_t targetLength)
{
   T *broadcastedData = new T[targetLength];
   std::size_t curLength = ConvertShapeToLength(shape);
   BroadcastTensor<T>(data, curLength, shape, targetShape, broadcastedData);
   return broadcastedData;
}

template <typename T>
T *UnidirectionalBroadcast(const T *data, const std::vector<std::size_t> &shape,
                           const std::vector<std::size_t> &targetShape)
{
   if (shape.size() < targetShape.size()) {
      std::size_t targetSize = targetShape.size();
      std::vector<std::size_t> newShape(targetSize, 1);
      std::size_t offset = targetSize - shape.size();
      std::copy(shape.begin(), shape.end(), newShape.begin() + offset);
      return CreateBroadcastTensor(data, newShape, targetShape, ConvertShapeToLength(targetShape));
   }
   return CreateBroadcastTensor(data, shape, targetShape, ConvertShapeToLength(targetShape));
}
