import { Asset } from 'expo-asset'
import * as FileSystem from 'expo-file-system'
import { loadTensorflowModel, type TfliteModel } from 'react-native-fast-tflite'

export type ModelStatus = {
  ready: boolean
  modelPath: string
  labels: string[]
  inputShape: number[]
  outputShape: number[]
  inputType: string
  reason?: string
}

export type LocalInferenceAdapter = {
  status: ModelStatus
  infer: (features: Float32Array) => Promise<{ label: string; confidence: number; index: number }>
}

function readFloat32(buffer: ArrayBuffer) {
  return new Float32Array(buffer)
}

function labelsFromValue(value: unknown) {
  if (Array.isArray(value)) return value.map(String)
  if (value && typeof value === 'object') return Object.values(value as Record<string, unknown>).map(String)
  return []
}

export async function loadLocalModel(): Promise<LocalInferenceAdapter> {
  const modelAsset = Asset.fromModule(require('../assets/models/psl_model.tflite'))
  const labelsAsset = Asset.fromModule(require('../assets/models/class_map.json'))
  await Promise.all([modelAsset.downloadAsync(), labelsAsset.downloadAsync()])
  const labelsText = await FileSystem.readAsStringAsync(labelsAsset.localUri ?? labelsAsset.uri)
  const labels = labelsFromValue(JSON.parse(labelsText))
  const model = await loadTensorflowModel(require('../assets/models/psl_model.tflite'), [])
  const status: ModelStatus = {
    ready: true,
    modelPath: modelAsset.localUri ?? modelAsset.uri,
    labels,
    inputShape: model.inputs[0]?.shape ?? [],
    outputShape: model.outputs[0]?.shape ?? [],
    inputType: model.inputs[0]?.dataType ?? 'unknown',
  }
  return { status, infer: createInference(model, labels) }
}

function createInference(model: TfliteModel, labels: string[]) {
  return async (features: Float32Array) => {
    const input = new Float32Array(features)
    const outputs = await model.run([input.buffer])
    const output = outputs[0]
    if (!output) throw new Error('مدل خروجی تولید نکرد')
    const scores = readFloat32(output)
    let index = 0
    for (let i = 1; i < scores.length; i += 1) {
      const score = scores[i] ?? Number.NEGATIVE_INFINITY
      const best = scores[index] ?? Number.NEGATIVE_INFINITY
      if (score > best) index = i
    }
    const rawConfidence = scores[index] ?? 0
    const confidence = rawConfidence > 1 ? rawConfidence / 100 : rawConfidence
    return { index, confidence: Math.max(0, Math.min(1, confidence)), label: labels[index] ?? `کلاس ${index + 1}` }
  }
}
