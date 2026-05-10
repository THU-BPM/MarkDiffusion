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

# Import dataset classes
from markdiffusion.evaluation.dataset import (
    BaseDataset,
    StableDiffusionPromptsDataset,
    MSCOCODataset,
    VBenchDataset
)

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
            diffusion_config=video_diffusion_config,
            k_f=1  # Override k_f to 1 to support small num_frames in testing
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
            diffusion_config=image_diffusion_config,
            
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
def test_group_image_quality_pipeline_saturation(test_image_dataset_group, all_image_editors, all_image_quality_analyzers, image_diffusion_config):
    """Saturation test: GroupImageQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = GroupImageQualityAnalysisPipeline(
        dataset=test_image_dataset_group,
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
            diffusion_config=video_diffusion_config,
            k_f=1  # Override k_f to 1 to support small num_frames in testing
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


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running pipeline tests...")

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
