# embeddings-similarity
![CI](https://github.com/FerrowAI/embeddings-similarity/actions/workflows/ci.yml/badge.svg)

Vector math and in-memory embedding search — cosine/dot/euclidean,
normalization, heap-based top-K, batch top-K, seeded k-means-lite
clustering. Zero runtime dependencies, strict TypeScript.

## Quickstart

```ts
import { cosineSimilarity, topK, kmeansLite } from "embeddings-similarity";

cosineSimilarity([1, 0, 0], [0.9, 0.1, 0]); // => 0.9939...

const vectors = [
  { vector: [1, 0, 0], metadata: { id: "east" } },
  { vector: [0, 1, 0], metadata: { id: "north" } },
  { vector: [0.95, 0.05, 0], metadata: { id: "near-east" } },
];

topK([1, 0, 0], vectors, 2, "cosine");
// => [{ index: 0, score: 1, metadata: {...} }, { index: 2, score: 0.998..., metadata: {...} }]

kmeansLite([[0, 0], [0.1, 0.1], [10, 10], [10.1, 9.9]], { k: 2, seed: 7 });
// => { centroids: [...], assignments: [...] }
```

## API

- **`cosineSimilarity(a, b): number`** — in `[-1, 1]`; returns `0` if either vector has zero magnitude.
- **`dot(a, b): number`** — raw dot product.
- **`euclideanDistance(a, b): number`** — L2 distance.
- **`magnitude(a): number`** — L2 norm.
- **`normalize(a): Vector`** — unit-length copy; zero vectors returned unchanged.
- **`topK(query, vectors, k, metric?): TopKResult[]`** — `metric` is `"cosine"` (default), `"dot"`, or `"euclidean"`. Uses a bounded min-heap of size `k`, not a full sort.
- **`batchTopK(queries, vectors, k, metric?): TopKResult[][]`** — `topK` for each query.
- **`kmeansLite(vectors, options): KMeansResult`** — `options: { k, iterations?, seed? }`. Fixed iteration count (default 10), seeded deterministic init (default seed 42).
- **`class DimensionMismatchError extends Error`** — thrown by any comparison given vectors of different lengths.

### Types

```ts
type Vector = number[];
type SimilarityMetric = "cosine" | "dot" | "euclidean";
interface VectorEntry<T> { vector: Vector; metadata?: T; }
interface TopKResult<T> { index: number; score: number; metadata?: T; }
interface KMeansOptions { k: number; iterations?: number; seed?: number; }
interface KMeansResult { centroids: Vector[]; assignments: number[]; }
```

## Limits

- **Brute-force exact search.** `topK`/`batchTopK` scan every candidate
  vector — fine up to roughly 100k vectors on a single machine, but there
  is no indexing (HNSW, IVF, etc). Beyond that scale, use a real vector
  database.
- `kmeansLite` runs a fixed iteration count with no convergence check —
  it is for quick exploratory clustering, not a tuned clustering library.
  Empty clusters keep their previous centroid rather than reseeding.
- All vector-pair functions throw `DimensionMismatchError` on length
  mismatch rather than silently truncating or padding.

---
Part of the [ferrow-toolkit](https://github.com/FerrowAI/ferrow-toolkit) collection · Sponsored by [Ferrow](https://ferrow.ai)
