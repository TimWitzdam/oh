import type { ModelSpec } from '../catalog';
import { OnnxClassifierDetector } from './onnx';
import { TorchDetector } from './torch';
import type { Detector } from './types';

export function createDetector(spec: ModelSpec): Detector {
  switch (spec.kind) {
    case 'onnx-classifier':
      return new OnnxClassifierDetector(spec);
    case 'torch-classifier':
      return new TorchDetector(spec);
  }
}

export type { Detector, ScoreUpdate } from './types';