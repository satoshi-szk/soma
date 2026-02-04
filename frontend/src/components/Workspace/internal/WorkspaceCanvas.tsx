import { Circle, Group, Layer, Line, Rect, Stage, Image as KonvaImage, Text } from 'react-konva'
import { AUTOMATION_LANE_HEIGHT } from '../../../app/constants'
import { useWorkspaceController } from './useWorkspaceController'

type WorkspaceController = ReturnType<typeof useWorkspaceController>

const hexToRgba = (hex: string, alpha: number) => {
  const cleaned = hex.trim().replace('#', '')
  if (!/^[0-9a-fA-F]{6}$/.test(cleaned)) {
    return `rgba(248, 209, 154, ${alpha})`
  }
  const r = parseInt(cleaned.slice(0, 2), 16)
  const g = parseInt(cleaned.slice(2, 4), 16)
  const b = parseInt(cleaned.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

type Props = {
  controller: WorkspaceController
}

export function WorkspaceCanvas({ controller }: Props) {
  const {
    stageSize,
    pan,
    contentOffset,
    scale,
    previewImage,
    preview,
    viewportImages,
    viewportPositionsByKey,
    buildViewportKey,
    unselectedPartials,
    renderPartials,
    selectionBox,
    playbackPosition,
    rulerHeight,
    automationTop,
    automationContentTop,
    rulerWidth,
    partials,
    playheadIntersections,
    freqRulerMarks,
    timeMarks,
    timeToX,
    freqToY,
    ampToLaneY,
    onStageWheel,
    handleStageMouseDown,
    handleStageMouseMove,
    handleStageMouseUp,
    handleStageClick,
    handleEndpointDragMove,
    handleEndpointDragEnd,
  } = controller

  return (
    <Stage
      width={stageSize.width}
      height={stageSize.height}
      onWheel={onStageWheel}
      onMouseDown={handleStageMouseDown}
      onMouseMove={handleStageMouseMove}
      onMouseUp={handleStageMouseUp}
      onClick={handleStageClick}
    >
      <Layer>
        <Group x={pan.x + contentOffset.x} y={pan.y + contentOffset.y} scaleX={scale.x} scaleY={scale.y}>
          {previewImage ? (
            <KonvaImage image={previewImage} width={preview?.width} height={preview?.height} opacity={0.85} />
          ) : null}
          {viewportImages.length > 0
            ? viewportImages.map((item) => {
                const position = viewportPositionsByKey.get(buildViewportKey(item.preview))
                if (!position) return null
                return (
                  <KonvaImage
                    key={`${item.preview.time_start}-${item.preview.time_end}-${item.preview.freq_min}-${item.preview.freq_max}`}
                    image={item.image}
                    x={position.x}
                    y={position.y}
                    width={position.width}
                    height={position.height}
                    opacity={0.95}
                  />
                )
              })
            : null}
        </Group>
      </Layer>
      <Layer>
        <Group x={pan.x + contentOffset.x} y={pan.y + contentOffset.y} scaleX={scale.x} scaleY={scale.y}>
          {unselectedPartials.map((partial) => (
            <Line
              key={partial.id}
              points={partial.points.flatMap((point) => [timeToX(point.time), freqToY(point.freq)])}
              stroke={hexToRgba(partial.color, partial.is_muted ? 0.25 : 0.6)}
              strokeWidth={3}
              strokeScaleEnabled={false}
              lineCap="round"
              lineJoin="round"
              listening={false}
            />
          ))}
          {renderPartials.map((partial) => (
            <Line
              key={partial.id}
              points={partial.points.flatMap((point) => [timeToX(point.time), freqToY(point.freq)])}
              stroke={hexToRgba(partial.color, partial.is_muted ? 0.35 : 0.95)}
              strokeWidth={3}
              strokeScaleEnabled={false}
              lineCap="round"
              lineJoin="round"
            />
          ))}
          {renderPartials.length === 1 && renderPartials[0].points.length >= 2 ? (
            <>
              {[0, renderPartials[0].points.length - 1].map((index) => (
                <Circle
                  key={index}
                  x={timeToX(renderPartials[0].points[index].time)}
                  y={freqToY(renderPartials[0].points[index].freq)}
                  radius={4}
                  fill={hexToRgba(renderPartials[0].color, 0.95)}
                  draggable
                  onDragMove={(event) => {
                    const { x, y } = event.target.position()
                    handleEndpointDragMove({ partial: renderPartials[0], index, position: { x, y } })
                  }}
                  onDragEnd={(event) => {
                    const { x, y } = event.target.position()
                    handleEndpointDragEnd({ partial: renderPartials[0], index, position: { x, y } })
                  }}
                />
              ))}
            </>
          ) : null}
        </Group>
        {selectionBox ? (
          <Rect x={selectionBox.x} y={selectionBox.y} width={selectionBox.w} height={selectionBox.h} stroke="#f59f8b" dash={[4, 4]} />
        ) : null}
        {preview ? (
          <Line
            points={[
              pan.x + contentOffset.x + timeToX(playbackPosition) * scale.x,
              rulerHeight,
              pan.x + contentOffset.x + timeToX(playbackPosition) * scale.x,
              stageSize.height,
            ]}
            stroke="rgba(247, 245, 242, 0.8)"
            strokeWidth={1}
          />
        ) : null}
        {playheadIntersections.map((intersection) => (
          <Group key={`playhead-note-${intersection.id}`} x={intersection.x + 6} y={intersection.y - 8}>
            <Rect
              width={Math.max(32, intersection.text.length * 6 + 8)}
              height={14}
              fill="rgba(8, 12, 18, 0.72)"
              cornerRadius={3}
              listening={false}
            />
            <Text
              x={4}
              y={2}
              text={intersection.text}
              fontSize={10}
              fill="rgba(245, 247, 250, 0.98)"
              fontFamily="monospace"
              listening={false}
            />
          </Group>
        ))}
      </Layer>
      <Layer>
        <Rect x={0} y={0} width={stageSize.width} height={rulerHeight} fill="rgba(12, 18, 30, 0.7)" />
        <Line points={[0, rulerHeight, stageSize.width, rulerHeight]} stroke="rgba(248, 209, 154, 0.35)" />
        {timeMarks.map((time) => {
          const x = pan.x + contentOffset.x + timeToX(time) * scale.x
          if (x < 0 || x > stageSize.width) return null
          return (
            <Group key={time}>
              <Line points={[x, rulerHeight - 6, x, rulerHeight]} stroke="rgba(248, 209, 154, 0.6)" />
              <Text
                x={x + 4}
                y={2}
                text={`${time.toFixed(time < 1 ? 2 : 1)}s`}
                fontSize={9}
                fill="rgba(248, 209, 154, 0.75)"
                fontFamily="monospace"
              />
            </Group>
          )
        })}
        <Rect
          x={0}
          y={automationTop}
          width={stageSize.width}
          height={AUTOMATION_LANE_HEIGHT}
          fill="rgba(10, 14, 20, 0.82)"
        />
        <Line points={[0, automationTop, stageSize.width, automationTop]} stroke="rgba(248, 209, 154, 0.25)" strokeWidth={1} />
        <Text
          x={12}
          y={automationTop + 6}
          text="Partial Amplitude"
          fontSize={9}
          fill="rgba(248, 209, 154, 0.7)"
          fontFamily="monospace"
        />
        <Group x={pan.x + contentOffset.x} y={automationContentTop} scaleX={scale.x}>
          {partials.map((partial) => (
            <Line
              key={`amp-${partial.id}`}
              points={partial.points.flatMap((point) => [timeToX(point.time), ampToLaneY(point.amp)])}
              stroke={hexToRgba(partial.color, partial.is_muted ? 0.25 : 0.85)}
              strokeWidth={3}
              strokeScaleEnabled={false}
            />
          ))}
        </Group>
        <Group x={stageSize.width - rulerWidth} y={pan.y + contentOffset.y} scaleY={scale.y}>
          {freqRulerMarks.map((mark) => {
            const y = freqToY(mark.freq)
            return (
              <Group key={mark.freq}>
                <Line points={[0, y, 8, y]} stroke="rgba(248, 209, 154, 0.5)" strokeWidth={1} />
                <Text
                  x={12}
                  y={y}
                  text={mark.label}
                  fontSize={10}
                  fill="rgba(248, 209, 154, 0.8)"
                  fontFamily="monospace"
                  scaleY={1 / scale.y}
                  offsetY={5}
                />
              </Group>
            )
          })}
        </Group>
      </Layer>
    </Stage>
  )
}
