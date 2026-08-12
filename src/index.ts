/**
 * embeddings-similarity
 *
 * Vector math and in-memory search for embedding vectors: cosine/dot/
 * euclidean distance, normalization, top-K search via a bounded heap
 * (not a full sort), batch top-K, and a fixed-iteration seeded k-means
 * for lightweight clustering. Brute-force exact search — fine up to
 * roughly 100k vectors; use a real vector DB (pgvector, Pinecone, etc)
 * beyond that. Zero runtime dependencies.
 */

export type Vector = number[];

export class DimensionMismatchError extends Error {
  constructor(dimA: number, dimB: number) {
    super(`embeddings-similarity: dimension mismatch (${dimA} vs ${dimB})`);
    this.name = "DimensionMismatchError";
  }
}

function assertSameDimension(a: Vector, b: Vector): void {
  if (a.length !== b.length) {
    throw new DimensionMismatchError(a.length, b.length);
  }
}

/** Dot product of two equal-length vectors. */
export function dot(a: Vector, b: Vector): number {
  assertSameDimension(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/** L2 (Euclidean) norm of a vector. */
export function magnitude(a: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * a[i];
  return Math.sqrt(sum);
}

/**
 * Cosine similarity between two vectors, in [-1, 1]. Returns 0 if either
 * vector has zero magnitude (rather than dividing by zero / NaN).
 */
export function cosineSimilarity(a: Vector, b: Vector): number {
  assertSameDimension(a, b);
  const magA = magnitude(a);
  const magB = magnitude(b);
  if (magA === 0 || magB === 0) return 0;
  return dot(a, b) / (magA * magB);
}

/** Euclidean (L2) distance between two vectors. */
export function euclideanDistance(a: Vector, b: Vector): number {
  assertSameDimension(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/** Return a new unit-length (L2-normalized) copy of `a`. Zero vectors are returned unchanged. */
export function normalize(a: Vector): Vector {
  const mag = magnitude(a);
  if (mag === 0) return a.slice();
  return a.map((x) => x / mag);
}

export type SimilarityMetric = "cosine" | "dot" | "euclidean";

function scoreFn(metric: SimilarityMetric): (a: Vector, b: Vector) => number {
  if (metric === "cosine") return cosineSimilarity;
  if (metric === "dot") return dot;
  return (a, b) => -euclideanDistance(a, b); // higher score = closer, so negate distance
}

export interface VectorEntry<T = unknown> {
  vector: Vector;
  metadata?: T;
}

export interface TopKResult<T = unknown> {
  index: number;
  score: number;
  metadata?: T;
}

/**
 * Min-heap keyed by `.score`, capped at `capacity`. Used to find the top-K
 * highest-scoring entries without sorting the full candidate set.
 */
class BoundedMinHeap<T> {
  private items: { score: number; value: T }[] = [];
  constructor(private capacity: number) {}

  get size(): number {
    return this.items.length;
  }

  peekMin(): number {
    return this.items[0]?.score ?? -Infinity;
  }

  push(score: number, value: T): void {
    if (this.items.length < this.capacity) {
      this.items.push({ score, value });
      this.bubbleUp(this.items.length - 1);
      return;
    }
    if (score <= this.items[0].score) return; // worse than current min, skip
    this.items[0] = { score, value };
    this.bubbleDown(0);
  }

  toSortedArray(): { score: number; value: T }[] {
    return [...this.items].sort((a, b) => b.score - a.score);
  }

  private bubbleUp(i: number): void {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.items[parent].score <= this.items[i].score) break;
      [this.items[parent], this.items[i]] = [this.items[i], this.items[parent]];
      i = parent;
    }
  }

  private bubbleDown(i: number): void {
    const n = this.items.length;
    while (true) {
      const left = 2 * i + 1;
      const right = 2 * i + 2;
      let smallest = i;
      if (left < n && this.items[left].score < this.items[smallest].score) smallest = left;
      if (right < n && this.items[right].score < this.items[smallest].score) smallest = right;
      if (smallest === i) break;
      [this.items[smallest], this.items[i]] = [this.items[i], this.items[smallest]];
      i = smallest;
    }
  }
}

/**
 * Find the top-K most similar vectors to `query` among `vectors`, using a
 * bounded min-heap of size K rather than sorting all candidates.
 */
export function topK<T = unknown>(
  query: Vector,
  vectors: VectorEntry<T>[],
  k: number,
  metric: SimilarityMetric = "cosine"
): TopKResult<T>[] {
  if (k <= 0) return [];
  const score = scoreFn(metric);
  const heap = new BoundedMinHeap<{ index: number; metadata?: T }>(k);

  vectors.forEach((entry, index) => {
    const s = score(query, entry.vector);
    heap.push(s, { index, metadata: entry.metadata });
  });

  return heap.toSortedArray().map(({ score: s, value }) => ({
    index: value.index,
    score: s,
    metadata: value.metadata,
  }));
}

/**
 * Run `topK` for multiple queries against the same vector set.
 */
export function batchTopK<T = unknown>(
  queries: Vector[],
  vectors: VectorEntry<T>[],
  k: number,
  metric: SimilarityMetric = "cosine"
): TopKResult<T>[][] {
  return queries.map((q) => topK(q, vectors, k, metric));
}

export interface KMeansOptions {
  k: number;
  /** Fixed iteration count (no convergence-based early exit). Default 10. */
  iterations?: number;
  /** Seed for deterministic centroid initialization. Default 42. */
  seed?: number;
}

export interface KMeansResult {
  centroids: Vector[];
  /** Cluster assignment (centroid index) for each input vector, in order. */
  assignments: number[];
}

/** Simple deterministic PRNG (mulberry32) for seeded, reproducible init. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Lightweight k-means clustering: fixed iteration count, seeded
 * pseudo-random centroid initialization (no convergence check). Intended
 * for quick exploratory clustering of small-to-medium vector sets, not as
 * a substitute for a tuned clustering library.
 */
export function kmeansLite(vectors: Vector[], options: KMeansOptions): KMeansResult {
  const { k } = options;
  const iterations = options.iterations ?? 10;
  const seed = options.seed ?? 42;

  if (vectors.length === 0) return { centroids: [], assignments: [] };
  if (k <= 0) throw new RangeError("kmeansLite: k must be > 0");
  if (k > vectors.length) throw new RangeError("kmeansLite: k cannot exceed the number of vectors");

  const dim = vectors[0].length;
  for (const v of vectors) assertSameDimension(v, vectors[0]);

  const rand = mulberry32(seed);

  // Seeded random selection of k distinct initial centroids (no replacement).
  const pool = vectors.map((_, i) => i);
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  let centroids: Vector[] = pool.slice(0, k).map((i) => vectors[i].slice());

  let assignments = new Array<number>(vectors.length).fill(0);

  for (let iter = 0; iter < iterations; iter++) {
    // Assign step
    for (let i = 0; i < vectors.length; i++) {
      let bestDist = Infinity;
      let bestCentroid = 0;
      for (let c = 0; c < centroids.length; c++) {
        const d = euclideanDistance(vectors[i], centroids[c]);
        if (d < bestDist) {
          bestDist = d;
          bestCentroid = c;
        }
      }
      assignments[i] = bestCentroid;
    }

    // Update step
    const sums = Array.from({ length: k }, () => new Array<number>(dim).fill(0));
    const counts = new Array<number>(k).fill(0);
    for (let i = 0; i < vectors.length; i++) {
      const c = assignments[i];
      counts[c]++;
      for (let d = 0; d < dim; d++) sums[c][d] += vectors[i][d];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) continue; // keep previous centroid if cluster is empty
      centroids[c] = sums[c].map((s) => s / counts[c]);
    }
  }

  return { centroids, assignments };
}
