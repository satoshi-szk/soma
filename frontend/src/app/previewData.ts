import type { SpectrogramPreview } from './types'

const toNumberArray = (buffer: ArrayBuffer): number[] => Array.from(new Uint8Array(buffer))

export const ensurePreviewData = async (preview: SpectrogramPreview): Promise<SpectrogramPreview> => {
  if (preview.data && preview.data.length > 0) {
    return preview
  }
  if (!preview.data_path) {
    return { ...preview, data: preview.data ?? [] }
  }
  try {
    const response = await fetch(preview.data_path, { cache: 'no-store' })
    if (!response.ok) {
      console.warn('[PreviewData] fetch failed:', response.status, response.statusText)
      return { ...preview, data: preview.data ?? [] }
    }
    const buffer = await response.arrayBuffer()
    const data = toNumberArray(buffer)
    if (preview.data_length && data.length !== preview.data_length) {
      console.warn('[PreviewData] data length mismatch:', data.length, 'expected', preview.data_length)
    }
    return { ...preview, data }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    console.warn('[PreviewData] fetch error:', message)
    return { ...preview, data: preview.data ?? [] }
  }
}
