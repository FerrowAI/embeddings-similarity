const {
  cosineSimilarity,
  dot,
  euclideanDistance,
  normalize,
  topK,
  batchTopK,
  kmeansLite,
  DimensionMismatchError,
} = require("../dist/index.js");

const a = [1, 0, 0];
const b = [0.9, 0.1, 0];
console.log("cosine(a,b):", cosineSimilarity(a, b).toFixed(4));
console.log("dot(a,b):", dot(a, b));
console.log("euclidean(a,b):", euclideanDistance(a, b).toFixed(4));
console.log("normalize([3,4]):", normalize([3, 4]));

const vectors = [
  { vector: [1, 0, 0], metadata: { id: "east" } },
  { vector: [0, 1, 0], metadata: { id: "north" } },
  { vector: [0.95, 0.05, 0], metadata: { id: "near-east" } },
  { vector: [-1, 0, 0], metadata: { id: "west" } },
  { vector: [0, 0, 1], metadata: { id: "up" } },
];

const query = [1, 0, 0];
console.log("topK(query, 3, cosine):", topK(query, vectors, 3, "cosine"));

const batch = batchTopK([[1, 0, 0], [0, 1, 0]], vectors, 2, "cosine");
console.log("batchTopK result count:", batch.length, "each k:", batch[0].length);

try {
  cosineSimilarity([1, 2], [1, 2, 3]);
} catch (err) {
  console.log("dimension mismatch caught:", err instanceof DimensionMismatchError, err.message);
}

const clusterVectors = [
  [0, 0], [0.1, 0.1], [0.2, 0],
  [10, 10], [10.1, 9.9], [9.8, 10.2],
];
const { centroids, assignments } = kmeansLite(clusterVectors, { k: 2, iterations: 10, seed: 7 });
console.log("kmeans centroids:", centroids.map((c) => c.map((x) => Number(x.toFixed(2)))));
console.log("kmeans assignments:", assignments);
