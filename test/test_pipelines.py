"""
Comprehensive tests for MarkDiffusion evaluation pipelines and datasets.

This module tests:
1. Dataset classes (StableDiffusionPromptsDataset, MSCOCODataset, VBenchDataset)
2. Detection pipelines (WatermarkedMediaDetectionPipeline, UnWatermarkedMediaDetectionPipeline)
3. Image quality analysis pipelines (5 pipelines)
4. Video quality analysis pipeline

All tests use saturation testing with all available editors and analyzers.

Usage:
    # Test all pipelines and datasets
    pytest test/test_pipelines.py -v

    # Test specific components
    pytest test/test_pipelines.py -m dataset -v
    pytest test/test_pipelines.py -m detection -v
    pytest test/test_pipelines.py -m quality -v
"""

import pytest
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

# Import dataset classes
from markdiffusion.evaluation.dataset import (
    BaseDataset,
    StableDiffusionPromptsDataset,
    MSCOCODataset,
    VBenchDataset
)

from markdiffusion.watermark.base import BaseWatermark

# Import pipeline classes
from markdiffusion.evaluation.pipelines.detection import (
    WatermarkedMediaDetectionPipeline,
    UnWatermarkedMediaDetectionPipeline,
    DetectionPipelineReturnType
)

from markdiffusion.evaluation.pipelines.image_quality_analysis import (
    DirectImageQualityAnalysisPipeline,
    ReferencedImageQualityAnalysisPipeline,
    GroupImageQualityAnalysisPipeline,
    RepeatImageQualityAnalysisPipeline,
    ComparedImageQualityAnalysisPipeline,
    QualityPipelineReturnType,
    QualityComparisonResult 
)

from markdiffusion.evaluation.pipelines.video_quality_analysis import (
    DirectVideoQualityAnalysisPipeline,
    QualityPipelineReturnType as VideoQualityPipelineReturnType,
    QualityComparisonResult as VideoQualityComparisonResult
)


# ============================================================================
# Test Cases - Detection Pipelines (Saturation Tests)
# ============================================================================

@pytest.mark.pipeline
@pytest.mark.detection
@pytest.mark.slow
def test_watermarked_detection_pipeline_with_all_image_editors(test_image_dataset, all_image_editors, image_diffusion_config):
    """Saturation test: WatermarkedMediaDetectionPipeline with all image editors."""
    from markdiffusion.watermark.auto_watermark import AutoWatermark

    # Initialize pipeline
    pipeline = WatermarkedMediaDetectionPipeline(
        dataset=test_image_dataset,
        media_editor_list=all_image_editors,
        return_type=DetectionPipelineReturnType.SCORES
    )

   #  assert len(pipeline.media_editor_list) == len(all_image_editors)
    assert pipeline.dataset == test_image_dataset

    # Load a watermark algorithm (use TR as example)
    try:
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='markdiffusion/config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, list), "Evaluate should return a list"
        assert len(result) > 0, "Evaluate should return non-empty results"

        print(f"✓ WatermarkedMediaDetectionPipeline with all {len(all_image_editors)} image editors test passed")
        print(f"  - Pipeline evaluated successfully with {len(result)} results")

    except Exception as e:
        pytest.fail(f"Watermark loading or evaluation error: {e}")


@pytest.mark.pipeline
@pytest.mark.detection
@pytest.mark.slow
def test_unwatermarked_detection_pipeline_with_all_image_editors(test_image_dataset, all_image_editors, image_diffusion_config):
    """Saturation test: UnWatermarkedMediaDetectionPipeline with all image editors."""
    from markdiffusion.watermark.auto_watermark import AutoWatermark

    # Initialize pipeline
    pipeline = UnWatermarkedMediaDetectionPipeline(
        dataset=test_image_dataset,
        media_editor_list=all_image_editors,
        return_type=DetectionPipelineReturnType.SCORES
    )

    assert len(pipeline.media_editor_list) == len(all_image_editors)
    assert pipeline.dataset == test_image_dataset

    # Load a watermark algorithm (use TR as example)
    try:
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='markdiffusion/config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, list), "Evaluate should return a list"
        assert len(result) > 0, "Evaluate should return non-empty results"

        print(f"✓ UnWatermarkedMediaDetectionPipeline with all {len(all_image_editors)} image editors test passed")
        print(f"  - Pipeline evaluated successfully with {len(result)} results")

    except Exception as e:
        pytest.fail(f"Watermark loading or evaluation error: {e}")


@pytest.mark.pipeline
@pytest.mark.detection
@pytest.mark.video
@pytest.mark.slow
def test_detection_pipeline_with_all_video_editors(test_video_dataset, all_video_editors, video_diffusion_config):
    """Saturation test: Detection pipeline with all video editors."""
    from markdiffusion.watermark.auto_watermark import AutoWatermark
    
    pipeline = WatermarkedMediaDetectionPipeline(
        dataset=test_video_dataset,
        media_editor_list=all_video_editors,
        detector_type="bit_acc",
        return_type=DetectionPipelineReturnType.SCORES
    )

    assert len(pipeline.media_editor_list) == len(all_video_editors)
    assert pipeline.dataset == test_video_dataset

    try:
        from markdiffusion.watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'VideoShield',
            algorithm_config='markdiffusion/config/VideoShield.json',
            diffusion_config=video_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, list), "Evaluate should return a list"
        assert len(result) > 0, "Evaluate should return non-empty results"

        print(f"✓ Detection pipeline with all {len(all_video_editors)} video editors test passed")
        print(f"  - Pipeline evaluated successfully with {len(result)} results")

    except Exception as e:
        pytest.fail(f"Watermark loading or evaluation error: {e}")


# ============================================================================
# Test Cases - Base Watermark Pipelines
# ============================================================================
@pytest.mark.pipeline
def test_base_generate_unwatermarked_video_i2v_branch(monkeypatch):
    import markdiffusion.watermark.base as base_mod
    monkeypatch.setattr(base_mod, "get_pipeline_type", lambda pipe: base_mod.PIPELINE_TYPE_IMAGE_TO_VIDEO)
    monkeypatch.setattr(base_mod, "is_video_pipeline", lambda pipe: True)
    monkeypatch.setattr(base_mod, "is_i2v_pipeline", lambda pipe: True)
    monkeypatch.setattr(base_mod, "is_t2v_pipeline", lambda pipe: False)

    class DummyI2VPipe:
        def __call__(self, **kwargs):
            frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
            return SimpleNamespace(frames=[frames])

    class DummyConfig:
        pipe = DummyI2VPipe()
        image_size = (32, 32)
        num_frames = 4
        init_latents = None
        num_inference_steps = 2
        guidance_scale = 1.0
        gen_seed = 0
        gen_kwargs = {}

    class DummyWatermark(base_mod.BaseWatermark):
        def __init__(self, config):
            super().__init__(config)
        def get_data_for_visualize(self, *args, **kwargs):
            return {}
        def _detect_watermark_in_video(self, *args, **kwargs):
            return {"ok": True}

    wm = DummyWatermark(DummyConfig())
    input_image = Image.new("RGB", (32, 32))
    frames = wm.generate_unwatermarked_media(input_image)
    assert isinstance(frames, list)
    assert len(frames) == 4
    assert all(isinstance(f, Image.Image) for f in frames)


@pytest.mark.pipeline
def test_base_preprocess_media_video_numpy_and_torch(monkeypatch):
    import markdiffusion.watermark.base as base_mod
    monkeypatch.setattr(base_mod, "get_pipeline_type", lambda pipe: base_mod.PIPELINE_TYPE_TEXT_TO_VIDEO)
    monkeypatch.setattr(base_mod, "is_image_pipeline", lambda pipe: False)
    monkeypatch.setattr(base_mod, "is_video_pipeline", lambda pipe: True)

    class DummyConfig:
        pipe = object()
        num_frames = 4
        image_size = (16, 16)

    class DummyWatermark(base_mod.BaseWatermark):
        def __init__(self, cfg):
            super().__init__(cfg)
        def get_data_for_visualize(self, *args, **kwargs):
            return {}
        def _detect_watermark_in_video(self, *args, **kwargs):
            return {"ok": True}

    wm = DummyWatermark(DummyConfig())

    # 1) numpy 4D: (F,H,W,C)
    arr = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    frames = wm._preprocess_media_for_detection(arr)
    assert isinstance(frames, list)
    assert len(frames) == 4
    assert all(isinstance(f, Image.Image) for f in frames)

    # 2) torch 5D: (B,C,F,H,W)
    t5 = torch.zeros(1, 3, 4, 16, 16)
    frames = wm._preprocess_media_for_detection(t5)
    assert isinstance(frames, list)
    assert len(frames) == 4

    # 3) torch 4D: (F,C,H,W)
    t4 = torch.zeros(4, 3, 16, 16)
    frames = wm._preprocess_media_for_detection(t4)
    assert isinstance(frames, list)
    assert len(frames) == 4


# ============================================================================
# Test Cases - Image Quality Analysis Pipelines (Saturation Tests)
# ============================================================================

@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.slow
def test_direct_image_quality_pipeline_saturation(test_image_dataset, all_image_editors, all_image_quality_analyzers, image_diffusion_config):
    """Saturation test: DirectImageQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = DirectImageQualityAnalysisPipeline(
        dataset=test_image_dataset,
        watermarked_image_editor_list=all_image_editors,
        unwatermarked_image_editor_list=all_image_editors,
        analyzers=all_image_quality_analyzers['direct'],
        return_type=QualityPipelineReturnType.FULL
    )

    assert len(pipeline.watermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.unwatermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.analyzers) == len(all_image_quality_analyzers['direct'])
    
    try:
        from markdiffusion.watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='markdiffusion/config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, QualityComparisonResult), "Evaluate should return QualityComparisonResult"

        print(f"✓ DirectImageQualityAnalysisPipeline saturation test passed")
        print(f"  - {len(all_image_editors)} editors per image type")
        print(f"  - {len(all_image_quality_analyzers['direct'])} analyzers")

    except Exception as e:
        pytest.fail(f"Watermark loading or evaluation error: {e}")
    

    


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.slow
def test_referenced_image_quality_pipeline_saturation(test_image_dataset, all_image_editors, all_image_quality_analyzers, image_diffusion_config):
    """Saturation test: ReferencedImageQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = ReferencedImageQualityAnalysisPipeline(
        dataset=test_image_dataset,
        watermarked_image_editor_list=all_image_editors,
        unwatermarked_image_editor_list=all_image_editors,
        analyzers=all_image_quality_analyzers['referenced'],
        unwatermarked_image_source='generated',
        reference_image_source='natural',
        return_type=QualityPipelineReturnType.FULL
    )

    assert len(pipeline.watermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.unwatermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.analyzers) == len(all_image_quality_analyzers['referenced'])
    
    try:
        from markdiffusion.watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='markdiffusion/config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, QualityComparisonResult), "Evaluate should return QualityComparisonResult"

        print(f"✓ ReferencedImageQualityAnalysisPipeline saturation test passed")
        print(f"  - {len(all_image_editors)} editors per image type")
        print(f"  - {len(all_image_quality_analyzers['referenced'])} analyzers")

    except Exception as e:
        pytest.fail(f"Watermark loading or evaluation error: {e}")




@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.slow
def test_group_image_quality_pipeline_saturation(test_image_dataset, all_image_editors, all_image_quality_analyzers, image_diffusion_config):
    """Saturation test: GroupImageQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = GroupImageQualityAnalysisPipeline(
        dataset=test_image_dataset,
        watermarked_image_editor_list=all_image_editors,
        unwatermarked_image_editor_list=all_image_editors,
        analyzers=all_image_quality_analyzers['group'],
        unwatermarked_image_source='generated',
        reference_image_source='natural',
        return_type=QualityPipelineReturnType.FULL
    )

    assert len(pipeline.watermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.unwatermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.analyzers) == len(all_image_quality_analyzers['group'])

    try:
        from markdiffusion.watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='markdiffusion/config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, QualityComparisonResult), "Evaluate should return QualityComparisonResult"

        print(f"✓ GroupImageQualityAnalysisPipeline saturation test passed")
        print(f"  - {len(all_image_editors)} editors per image type")
        print(f"  - {len(all_image_quality_analyzers['group'])} analyzers")

    except Exception as e:
        pytest.fail(f"Watermark loading or evaluation error: {e}")

    


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.slow
def test_repeat_image_quality_pipeline_saturation(test_image_dataset, all_image_editors, all_image_quality_analyzers, image_diffusion_config):
    """Saturation test: RepeatImageQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = RepeatImageQualityAnalysisPipeline(
        dataset=test_image_dataset,
        prompt_per_image=5,  # Small number for testing
        watermarked_image_editor_list=all_image_editors,
        unwatermarked_image_editor_list=all_image_editors,
        analyzers=all_image_quality_analyzers['repeat'],
        return_type=QualityPipelineReturnType.FULL
    )

    assert len(pipeline.watermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.unwatermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.analyzers) == len(all_image_quality_analyzers['repeat'])
    assert pipeline.prompt_per_image == 5
    
    try:
        from markdiffusion.watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='markdiffusion/config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, QualityComparisonResult), "Evaluate should return QualityComparisonResult"

        print(f"✓ RepeatImageQualityAnalysisPipeline saturation test passed")
        print(f"  - {len(all_image_editors)} editors per image type")
        print(f"  - {len(all_image_quality_analyzers['repeat'])} analyzers")

    except Exception as e:
        pytest.fail(f"Watermark loading or evaluation error: {e}")

    


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.slow
def test_compared_image_quality_pipeline_saturation(test_image_dataset, all_image_editors, all_image_quality_analyzers, image_diffusion_config):
    """Saturation test: ComparedImageQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = ComparedImageQualityAnalysisPipeline(
        dataset=test_image_dataset,
        watermarked_image_editor_list=all_image_editors,
        unwatermarked_image_editor_list=all_image_editors,
        analyzers=all_image_quality_analyzers['compared'],
        return_type=QualityPipelineReturnType.FULL
    )

    assert len(pipeline.watermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.unwatermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.analyzers) == len(all_image_quality_analyzers['compared'])
    
    try:
        from markdiffusion.watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='markdiffusion/config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, QualityComparisonResult), "Evaluate should return QualityComparisonResult"

        print(f"✓ ComparedImageQualityAnalysisPipeline saturation test passed")
        print(f"  - {len(all_image_editors)} editors per image type")
        print(f"  - {len(all_image_quality_analyzers['compared'])} analyzers")

    except Exception as e:
        pytest.fail(f"Watermark loading or evaluation error: {e}")




# ============================================================================
# Test Cases - Video Quality Analysis Pipeline (Saturation Test)
# ============================================================================

@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.video
@pytest.mark.slow
def test_video_quality_pipeline_saturation(test_video_dataset, all_video_editors, all_image_editors, all_video_quality_analyzers, video_diffusion_config):
    """Saturation test: DirectVideoQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = DirectVideoQualityAnalysisPipeline(
        dataset=test_video_dataset,
        watermarked_video_editor_list=all_video_editors,
        unwatermarked_video_editor_list=all_video_editors,
        watermarked_frame_editor_list=[],
        unwatermarked_frame_editor_list=[],
        analyzers=all_video_quality_analyzers,
        return_type=VideoQualityPipelineReturnType.FULL
    )

    assert len(pipeline.watermarked_video_editor_list) == len(all_video_editors)
    assert len(pipeline.unwatermarked_video_editor_list) == len(all_video_editors)
    assert len(pipeline.analyzers) == len(all_video_quality_analyzers)
    
    try:
        from markdiffusion.watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'VideoShield',
            algorithm_config='markdiffusion/config/VideoShield.json',
            diffusion_config=video_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, VideoQualityComparisonResult), "Evaluate should return QualityComparisonResult"

        print(f"✓ DirectVideoQualityAnalysisPipeline saturation test passed")
        print(f"  - {len(all_video_editors)} video editors per video type")
        print(f"  - {len(all_image_editors)} frame editors per video type")
        print(f"  - {len(all_video_quality_analyzers)} analyzers")

    except Exception as e:
        pytest.fail(f"Watermark loading or evaluation error: {e}")


# ============================================================================
# Test Cases - Success Rate Calculators
# ============================================================================

@pytest.mark.calculator
def test_fundamental_success_rate_calculator():
    """Test FundamentalSuccessRateCalculator basic functionality."""
    from markdiffusion.evaluation.tools.success_rate_calculator import (
        FundamentalSuccessRateCalculator,
        DetectionResult
    )
    
    calculator = FundamentalSuccessRateCalculator()
    
    # Test with boolean inputs
    watermarked = [True, True, False]
    non_watermarked = [False, False, True]
    metrics = calculator.calculate(watermarked, non_watermarked)
    assert 'TPR' in metrics
    
    # Test with DetectionResult objects
    watermarked_obj = [DetectionResult(True, 0.9), DetectionResult(True, 0.8)]
    non_watermarked_obj = [DetectionResult(False, 0.1), DetectionResult(False, 0.2)]
    metrics = calculator.calculate(watermarked_obj, non_watermarked_obj)
    assert 'TPR' in metrics
    assert 'TNR' in metrics
    assert 'F1' in metrics
    assert 'FPR' in metrics
    assert 'FNR' in metrics
    assert 'P' in metrics
    assert 'R' in metrics
    assert 'ACC' in metrics

    print("✓ FundamentalSuccessRateCalculator test passed")


@pytest.mark.calculator
def test_dynamic_threshold_calculator():
    """Test DynamicThresholdSuccessRateCalculator with different rules."""
    from markdiffusion.evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
    from markdiffusion.exceptions.exceptions import ConfigurationError
    
    # Test 'best' rule
    calc_best = DynamicThresholdSuccessRateCalculator(rule='best')
    watermarked = [0.9, 0.8, 0.7]
    non_watermarked = [0.3, 0.2, 0.1]
    metrics = calc_best.calculate(watermarked, non_watermarked)
    assert 'F1' in metrics
    
    # Test 'target_fpr' rule
    calc_fpr = DynamicThresholdSuccessRateCalculator(rule='target_fpr', target_fpr=0.1)
    metrics = calc_fpr.calculate(watermarked, non_watermarked)
    assert 'FPR' in metrics
    
    # Test reverse mode
    calc_reverse = DynamicThresholdSuccessRateCalculator(rule='best', reverse=True)
    metrics = calc_reverse.calculate([0.1, 0.2], [0.8, 0.9])
    assert 'TPR' in metrics
    
    # Test error handling
    try:
        DynamicThresholdSuccessRateCalculator(rule='invalid')
        assert False, "Should raise ConfigurationError"
    except ConfigurationError:
        pass
    
    try:
        DynamicThresholdSuccessRateCalculator(rule='target_fpr', target_fpr=1.5)
        assert False, "Should raise ConfigurationError"
    except ConfigurationError:
        pass
    
    # Test boolean rejection
    try:
        calc_best.calculate([True], [False])
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✓ DynamicThresholdSuccessRateCalculator test passed")

@pytest.mark.pipeline
@pytest.mark.detection
def test_unwatermarked_generated_branch_single_and_list():
    """Cover UnWatermarkedMediaDetectionPipeline._generate_or_retrieve_media 'generated' branch
    for both single Image and list[Image] return types."""

    class DummyDataset:
        def __init__(self, num_samples=3):
            self.num_samples = num_samples
            self.num_references = 0
        def get_prompt(self, index: int) -> str:
            return f"prompt-{index}"
   
    class DummyWatermark:
        def __init__(self, return_list: bool = False, img_size=(64, 64)):
            self.return_list = return_list
            self.img_size = img_size
        def generate_unwatermarked_media(self, input_data: str, **kwargs):
            img = Image.new("RGB", self.img_size, color="white")
            return [img, img] if self.return_list else img
        def detect_watermark_in_media(self, media, detector_type="l1_distance", **kwargs):
            # 返回包含 detector_type 对应键的字典（默认 l1_distance）
            return {
                "is_watermarked": False,
                "l1_distance": 0.123,
            }
    
    dataset = DummyDataset(num_samples=3)
    pipeline = UnWatermarkedMediaDetectionPipeline(
        dataset=dataset,
        media_editor_list=[],
        show_progress=False,
        detector_type="l1_distance",
        return_type=DetectionPipelineReturnType.SCORES,
        media_source_mode="generated",
    )

    wm_single = DummyWatermark(return_list=False)
    media_list_single = pipeline._generate_or_retrieve_media(0, wm_single)
    assert isinstance(media_list_single, list)
    assert len(media_list_single) >= 1
    assert isinstance(media_list_single[0], Image.Image)
    scores_single = pipeline.evaluate(watermark=wm_single, detection_kwargs={}, generation_kwargs={})
    assert isinstance(scores_single, list)
    assert len(scores_single) == dataset.num_samples
    assert all(isinstance(s, float) for s in scores_single)

    wm_list = DummyWatermark(return_list=True)
    media_list_list = pipeline._generate_or_retrieve_media(0, wm_list)
    assert isinstance(media_list_list, list)
    assert len(media_list_list) >= 2
    assert isinstance(media_list_list[0], Image.Image)
    scores_list = pipeline.evaluate(watermark=wm_list, detection_kwargs={}, generation_kwargs={})
    assert isinstance(scores_list, list)
    assert len(scores_list) == dataset.num_samples
    assert all(isinstance(s, float) for s in scores_list)

    print("✓ UnWatermarkedMediaDetectionPipeline generated branch (single and list) test passed")


# ============================================================================
# Test Cases - Image/Video Quality Pipeline _format_results and _store_results
# ============================================================================
@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.image
def test_image_quality_pipeline_format_results_full():
    """Test _format_results with FULL return type."""
    mock_dataset = MagicMock()
    mock_dataset.num_samples = 2
    mock_dataset.num_references = 0
    
    pipeline = DirectImageQualityAnalysisPipeline(
        dataset=mock_dataset,
        return_type=QualityPipelineReturnType.FULL,
        show_progress=False
    )
    
    result = QualityComparisonResult(
        store_path="/tmp/test",
        watermarked_quality_scores={'Analyzer1': [0.8, 0.9]},
        unwatermarked_quality_scores={'Analyzer1': [0.9, 0.95]},
        prompts=["prompt1", "prompt2"]
    )
    
    formatted = pipeline._format_results(result)
    assert isinstance(formatted, QualityComparisonResult)
    print("✓ Image _format_results FULL test passed")


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.image
def test_image_quality_pipeline_format_results_scores():
    """Test _format_results with SCORES return type."""
    
    mock_dataset = MagicMock()
    pipeline = DirectImageQualityAnalysisPipeline(
        dataset=mock_dataset,
        return_type=QualityPipelineReturnType.SCORES,
        show_progress=False
    )
    
    result = QualityComparisonResult(
        store_path="/tmp/test",
        watermarked_quality_scores={'Analyzer1': [0.8, 0.9]},
        unwatermarked_quality_scores={'Analyzer1': [0.9, 0.95]},
        prompts=["p1", "p2"]
    )
    
    formatted = pipeline._format_results(result)
    assert isinstance(formatted, dict)
    assert 'watermarked' in formatted
    assert 'unwatermarked' in formatted
    assert 'prompts' in formatted
    assert formatted['prompts'] == ["p1", "p2"]
    print("✓ Image _format_results SCORES test passed")


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.image
def test_image_quality_pipeline_format_results_mean_scores():
    """Test _format_results with MEAN_SCORES return type."""

    mock_dataset = MagicMock()
    pipeline = DirectImageQualityAnalysisPipeline(
        dataset=mock_dataset,
        return_type=QualityPipelineReturnType.MEAN_SCORES,
        show_progress=False
    )
    
    result = QualityComparisonResult(
        store_path="/tmp/test",
        watermarked_quality_scores={'Analyzer1': [0.8, 0.9]},
        unwatermarked_quality_scores={'Analyzer1': [0.9, 0.95]},
        prompts=["p1", "p2"]
    )
    
    formatted = pipeline._format_results(result)
    assert isinstance(formatted, dict)
    assert 'watermarked' in formatted
    assert isinstance(formatted['watermarked']['Analyzer1'], (float, np.floating))
    assert np.isclose(formatted['watermarked']['Analyzer1'], 0.85)
    print("✓ Image _format_results MEAN_SCORES test passed")


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.image
def test_image_quality_pipeline_format_results_edge_cases():
    """Test _format_results with edge cases for coverage."""
    
    mock_dataset = MagicMock()
    pipeline = DirectImageQualityAnalysisPipeline(
        dataset=mock_dataset,
        return_type=QualityPipelineReturnType.MEAN_SCORES,
        show_progress=False
    )
    
    # Test with non-list scores and empty list
    result = QualityComparisonResult(
        store_path="/tmp/test",
        watermarked_quality_scores={'A1': 0.85, 'A2': []},
        unwatermarked_quality_scores={'A1': 0.95, 'A2': []},
        prompts=[]
    )
    
    formatted = pipeline._format_results(result)
    assert formatted['watermarked']['A1'] == 0.85
    assert formatted['watermarked']['A2'] == []
    print("✓ Image _format_results edge cases test passed")


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.image
def test_image_quality_pipeline_store_results(tmp_path):
    """Test _store_results functionality."""
    from markdiffusion.evaluation.pipelines.image_quality_analysis import DatasetForEvaluation
    
    mock_dataset = MagicMock()
    mock_dataset.name = "test_dataset"
    mock_dataset.num_samples = 2
    
    pipeline = DirectImageQualityAnalysisPipeline(
        dataset=mock_dataset,
        store_path=str(tmp_path),
        show_progress=False
    )
    
    # Create mock images
    watermarked_img = Image.new('RGB', (64, 64), color='red')
    unwatermarked_img = Image.new('RGB', (64, 64), color='blue')
    
    prepared_dataset = DatasetForEvaluation(
        watermarked_images=[watermarked_img, watermarked_img],
        unwatermarked_images=[unwatermarked_img, unwatermarked_img],
        indexes=[0, 1],
        prompts=["prompt1", "prompt2"]
    )
    
    pipeline._store_results(prepared_dataset)
    
    # Verify files are created
    import os
    assert os.path.exists(os.path.join(tmp_path, f"DirectImageQualityAnalysisPipeline_test_dataset_watermarked_prompt_0.png"))
    assert os.path.exists(os.path.join(tmp_path, f"DirectImageQualityAnalysisPipeline_test_dataset_unwatermarked_prompt_1.png"))
    print("✓ Image _store_results test passed")


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.video
def test_video_quality_pipeline_format_results_full():
    """Test _format_results with FULL return type."""
    mock_dataset = MagicMock()
    mock_dataset.num_samples = 2
    mock_dataset.num_references = 0
    
    pipeline = DirectVideoQualityAnalysisPipeline(
        dataset=mock_dataset,
        return_type=VideoQualityPipelineReturnType.FULL,
        show_progress=False
    )
    
    result = QualityComparisonResult(
        store_path="/tmp/test",
        watermarked_quality_scores={'Analyzer1': [0.8, 0.9]},
        unwatermarked_quality_scores={'Analyzer1': [0.9, 0.95]},
        prompts=["prompt1", "prompt2"]
    )
    
    formatted = pipeline._format_results(result)
    assert isinstance(formatted, QualityComparisonResult)
    print("✓ Video _format_results FULL test passed")


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.video
def test_video_quality_pipeline_format_results_scores():
    """Test _format_results with SCORES return type."""
    
    mock_dataset = MagicMock()
    pipeline = DirectVideoQualityAnalysisPipeline(
        dataset=mock_dataset,
        return_type=VideoQualityPipelineReturnType.SCORES,
        show_progress=False
    )
    
    result = QualityComparisonResult(
        store_path="/tmp/test",
        watermarked_quality_scores={'Analyzer1': [0.8, 0.9]},
        unwatermarked_quality_scores={'Analyzer1': [0.9, 0.95]},
        prompts=["p1", "p2"]
    )
    
    formatted = pipeline._format_results(result)
    assert isinstance(formatted, dict)
    assert 'watermarked' in formatted
    assert 'unwatermarked' in formatted
    assert 'prompts' in formatted
    print("✓ Video _format_results SCORES test passed")


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.video
def test_video_quality_pipeline_format_results_mean_scores():
    """Test _format_results with MEAN_SCORES return type."""

    mock_dataset = MagicMock()
    pipeline = DirectVideoQualityAnalysisPipeline(
        dataset=mock_dataset,
        return_type=VideoQualityPipelineReturnType.MEAN_SCORES,
        show_progress=False
    )
    
    result = QualityComparisonResult(
        store_path="/tmp/test",
        watermarked_quality_scores={'Analyzer1': [0.8, 0.9]},
        unwatermarked_quality_scores={'Analyzer1': [0.9, 0.95]},
        prompts=["p1", "p2"]
    )
    
    formatted = pipeline._format_results(result)
    assert isinstance(formatted, dict)
    assert 'watermarked' in formatted
    assert isinstance(formatted['watermarked']['Analyzer1'], (float, np.floating))
    print("✓ Video _format_results MEAN_SCORES test passed")


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.video
def test_video_quality_pipeline_format_results_edge_cases():
    """Test _format_results with edge cases for coverage."""
    
    mock_dataset = MagicMock()
    pipeline = DirectVideoQualityAnalysisPipeline(
        dataset=mock_dataset,
        return_type=VideoQualityPipelineReturnType.MEAN_SCORES,
        show_progress=False
    )
    
    # Test with non-list scores
    result = QualityComparisonResult(
        store_path="/tmp/test",
        watermarked_quality_scores={'A1': 0.85, 'A2': []},
        unwatermarked_quality_scores={'A1': 0.95, 'A2': []},
        prompts=[]
    )
    
    formatted = pipeline._format_results(result)
    assert formatted['watermarked']['A1'] == 0.85
    assert formatted['watermarked']['A2'] == []
    print("✓ Video _format_results edge cases test passed")


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.video
def test_video_quality_pipeline_store_results(tmp_path):
    """Test _store_results method for coverage."""
    from markdiffusion.evaluation.pipelines.video_quality_analysis import (
        DirectVideoQualityAnalysisPipeline,
        DatasetForEvaluation
    )
    
    mock_dataset = MagicMock()
    mock_dataset.num_samples = 2
    mock_dataset.num_references = 0
    mock_dataset.name = "test_ds"
    
    store_path = str(tmp_path / "results")
    pipeline = DirectVideoQualityAnalysisPipeline(
        dataset=mock_dataset,
        store_path=store_path,
        show_progress=False
    )
    
    # Create mock frames
    frames1 = [Image.new('RGB', (32, 32), color=(i*80, i*80, i*80)) for i in range(2)]
    frames2 = [Image.new('RGB', (32, 32), color=(i*80, i*80, i*80)) for i in range(2)]
    
    prepared_dataset = DatasetForEvaluation()
    prepared_dataset.watermarked_videos = [frames1, frames2]
    prepared_dataset.unwatermarked_videos = [frames1, frames2]
    prepared_dataset.indexes = [0, 1]
    
    pipeline._store_results(prepared_dataset)
    
    # Verify files exist
    assert Path(store_path).exists()
    assert (Path(store_path) / f"{pipeline.__class__.__name__}_test_ds_watermarked_prompt0" / "frame_0.png").exists()
    assert (Path(store_path) / f"{pipeline.__class__.__name__}_test_ds_unwatermarked_prompt0" / "frame_0.png").exists()
    print("✓ Video _store_results test passed")


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.video
def test_video_quality_pipeline_store_results_with_references(tmp_path):
    """Test _store_results with references for coverage."""
    from markdiffusion.evaluation.pipelines.video_quality_analysis import (
        DirectVideoQualityAnalysisPipeline,
        DatasetForEvaluation
    )
    
    mock_dataset = MagicMock()
    mock_dataset.num_samples = 1
    mock_dataset.num_references = 1
    mock_dataset.name = "test_ref_ds"
    mock_dataset.get_reference = MagicMock(return_value=[Image.new('RGB', (32, 32))])
    
    store_path = str(tmp_path / "ref_results")
    pipeline = DirectVideoQualityAnalysisPipeline(
        dataset=mock_dataset,
        store_path=store_path,
        show_progress=False
    )
    
    frames = [Image.new('RGB', (32, 32)) for _ in range(2)]
    
    prepared_dataset = DatasetForEvaluation()
    prepared_dataset.watermarked_videos = [frames]
    prepared_dataset.unwatermarked_videos = [frames]
    prepared_dataset.reference_videos = [[Image.new('RGB', (32, 32))]]
    prepared_dataset.indexes = [0]
    
    pipeline._store_results(prepared_dataset)
    
    # Verify reference saved
    ref_dir = Path(store_path) / f"{pipeline.__class__.__name__}_test_ref_ds_reference_prompt0"
    assert ref_dir.exists()
    assert (ref_dir / "frame_0.png").exists()
    print("✓ Video _store_results with references test passed")


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running pipeline tests...")

    # Test calculators
    test_fundamental_success_rate_calculator()
    test_dynamic_threshold_calculator()
    
    # Test pipelines
    test_watermarked_detection_pipeline_with_all_image_editors()
    test_unwatermarked_detection_pipeline_with_all_image_editors()
    test_detection_pipeline_with_all_video_editors()
    test_direct_image_quality_pipeline_saturation()
    test_referenced_image_quality_pipeline_saturation()
    test_group_image_quality_pipeline_saturation()
    test_repeat_image_quality_pipeline_saturation()
    test_compared_image_quality_pipeline_saturation()
    test_video_quality_pipeline_saturation()
    print("\n✓ All basic tests completed successfully!")
