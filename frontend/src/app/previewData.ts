import type { SpectrogramPreview } from './types'

export const ensurePreviewData = async (preview: SpectrogramPreview): Promise<SpectrogramPreview> => {
  return { ...preview, data: preview.data ?? [] }
}
