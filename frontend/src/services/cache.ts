interface CacheEntry<T> {
  expiresAt: number;
  data: T;
}

const responseCache = new Map<string, CacheEntry<unknown>>();

export function getCached<T>(key: string): T | null {
  const record = responseCache.get(key);
  if (!record) {
    return null;
  }
  if (Date.now() > record.expiresAt) {
    responseCache.delete(key);
    return null;
  }
  return record.data as T;
}

export function setCached<T>(key: string, data: T, ttlMs: number): void {
  responseCache.set(key, {
    data,
    expiresAt: Date.now() + ttlMs
  });
}

export function clearCache(key?: string): void {
  if (key) {
    responseCache.delete(key);
    return;
  }
  responseCache.clear();
}
