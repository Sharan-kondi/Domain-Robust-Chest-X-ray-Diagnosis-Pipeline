# Model Card: Domain-Robust Chest X-ray Classifier

## Model Details

- **Architecture**: ResNet18 (ImageNet pretrained) with multi-task classification + bbox regression heads
- **Framework**: PyTorch + PyTorch Lightning
- **Domain Generalization**: CORAL loss alignment
- **Uncertainty**: MC Dropout (20 forward passes)
- **Training Data**: NIH ChestX-ray14 (6,000 images) + VinDr-CXR (4,000 images)
- **External Test Data**: Open-I / Indiana University (7,470 images — never seen during training)

## Intended Use

- **Primary**: Research demonstration of domain-robust medical image classification
- **Not intended for**: Clinical diagnosis or patient care decisions

## Shared Label Taxonomy

Atelectasis, Cardiomegaly, Consolidation, Effusion, Pneumonia, Pneumothorax, Nodule/Mass, No Finding

## Performance

### In-Distribution vs. External (Domain Gap)

*To be populated after training.*

### Subgroup Analysis

*To be populated after audit.*

## Limitations & Compute Trade-offs

- Training performed on consumer hardware (AMD Ryzen 7 4800H, 16GB RAM)
- GPU availability affects whether SSL pretraining (SimCLR) is feasible
- Subsample sizes chosen to fit RAM constraints — not representative of full dataset scale

## Ethical Considerations

- Model should **never** be used for autonomous clinical decisions
- Uncertainty routing ensures low-confidence predictions go to human review
- Subgroup audit identifies potential demographic performance disparities
