import { mapColor } from '../app/utils'

type RenderRequest = {
  type: 'render'
  key: string
  width: number
  height: number
  data: number[]
  brightness: number
  contrast: number
}

type RenderResponse =
  | { type: 'result'; key: string; bitmap: ImageBitmap }
  | { type: 'error'; key: string; message: string }

const renderViewport = (message: RenderRequest) => {
  const { key, width, height, data, brightness, contrast } = message
  if (width <= 0 || height <= 0) {
    return { type: 'error', key, message: 'Invalid viewport dimensions' } satisfies RenderResponse
  }
  const expected = width * height
  if (data.length !== expected) {
    return {
      type: 'error',
      key,
      message: `Viewport data length mismatch (expected ${expected}, got ${data.length})`,
    } satisfies RenderResponse
  }

  const canvas = new OffscreenCanvas(width, height)
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return { type: 'error', key, message: 'OffscreenCanvas context unavailable' } satisfies RenderResponse
  }
  const image = ctx.createImageData(width, height)
  for (let i = 0; i < data.length; i += 1) {
    const normalized = data[i] / 255
    const adjusted = Math.min(1, Math.max(0, (normalized - 0.5) * contrast + 0.5 + brightness))
    const value = Math.round(adjusted * 255)
    const color = mapColor(value)
    const offset = i * 4
    image.data[offset] = color[0]
    image.data[offset + 1] = color[1]
    image.data[offset + 2] = color[2]
    image.data[offset + 3] = 255
  }
  ctx.putImageData(image, 0, 0)
  const bitmap = canvas.transferToImageBitmap()
  return { type: 'result', key, bitmap } satisfies RenderResponse
}

const workerSelf = self as unknown as {
  postMessage: (message: unknown, transfer?: Transferable[]) => void
  onmessage: ((event: MessageEvent<RenderRequest>) => void) | null
}

workerSelf.onmessage = (event: MessageEvent<RenderRequest>) => {
  const payload = event.data
  if (!payload || payload.type !== 'render') return
  try {
    const response = renderViewport(payload)
    if (response.type === 'result') {
      workerSelf.postMessage(response, [response.bitmap])
    } else {
      workerSelf.postMessage(response)
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'viewport render failed'
    workerSelf.postMessage({ type: 'error', key: payload.key, message } satisfies RenderResponse)
  }
}
