# Topological Classification Network Demo

This repo contains code for a simple topological classification network on a subset of a publicly available industrial dataset. Our goal is to take various meshes from this database, inflict a defect upon some of them, then use topology and standard methods to detect which parts are flawed. 

## Breakdown of outcomes, deliverables

This repo contains some brief work on a classifier network using persistent homology to detect defects (holes, dents) in simple industrial manufacturing parts. Persistent homology is a process through which we can measure how topological features "persist" through the mesh as we move in specified directions, so our goal is to take this information and have a simple classifier evaluate whether the persistence indicated the part is flawed or not. Data was sourced from a publicly available database of collected meshes for industrial classification purposes. For this task, the following were constructed: 
- A small framework to repair and then deform each mesh, adding the desired defect by firing a ray at the mesh from outside towards a slightly off-center center of mass.
  - Dents are formed by taking the struck mesh, creating a new vertex, then pushing the vertex into the mesh's interior. The struck face is deleted, and new faces are created between the new vertex and the vertices of the old face.
  - Holes are formed by firing a ray straight through the mesh, then deleting both faces which the ray strikes. The edges of the hole which are closest are connected by new faces in order to re-close the mesh around the hole.
- A framework to extract persistent homology from meshes was constructed, using persistence from 6 directions around the mesh, as well as from above and below. This provides 8 barcodes, which for this task are fairly rich sources of information.
- A simple classifier network for persistent homology barcodes was created and trained, which can be found in this repo.
- A similar classifier network for meshes using Conv3D was created and can also be found in this repo.
  - Results indicate the topological network is significantly better on this task than the Conv3D network. Classification accuracy is improved by 6% (for the no-defect class) to a significant 15% jump in accuracy (from 84.3% to 97.7%) on objects with holes. On dents the improvement was a more modest 10%, but still remarkable.

## Data

We include training and validation data for the subset of the dataset we make use of, but the full dataset can be found here:

[MCBDataset](https://engineering.purdue.edu/cdesign/wp/a-large-scale-annotated-mechanical-components-benchmark-for-classification-and-retrieval-tasks-with-deep-neural-networks/#:~:text=introduce%20a%20large%2Dscale%20annotated,feature%20learning%20for%20mechanical%20components.)

We specifically use Dataset A. This dataset is a collected resource of annotated mechanical components, formatted in .obj files. Parts are collected from sources such as TraceParts, 3D Warehouse, and GrabCAD, then purified and annotated. For more information, see:

```bibtex
@article{zhang2020large,
  title={A Large-scale Annotated Mechanical Components Benchmark for Classification and Retrieval Tasks with Deep Neural Networks},
  author={Zhang, L. and Suchan, J.},
  journal={International Journal of Computer Vision},
  year={2020}
}
```

## Installation 

Clone the repo, then install the requirements:

```
pip install -r requirements.txt
```

Add any additional data desired from the above dataset to the train/test files, or proceed with the provided excerpt.

## Usage

To validate our networks, run:
```
python validate.py
```

To train a new PersNet classifier model, run:

```
python train.py
```

And to train a new Conv3D classifier model, run:

```
python train_conv.py
```

To see a brief demonstration of our method, run:
```
jupyter notebook
```
and check out our Demo.ipnyb.

## Results
Accuracy results for our PersNet and Conv3D classifier networks on a subset of the MCB dataset.

| Metric / Class | PersNet | Conv3DNet |
| :--- | :---: | :---: |
| **Overall Accuracy** | **96.23%** | **86.69%** |
| Accuracy [Normal] | 96.82% | 90.99% |
| Accuracy [Hole] | 97.70% | 84.28% |
| Accuracy [Dent] | 94.17% | 84.81% |

### Some conclusions and thoughts

- Persistent homology makes use of the mesh through aligning features directionally, so that the persistent features are lossless vis a vis resolution but not direction. This is why 8 distinct directions were used for this demo. However, it may be the case that even better results can be gleaned with a simplicial convolutional neural network built over the faces of the simplices, since the persistence is already so informative. The roadblocks for this are:
  - Tuning. SCNNs are high information but require some thought as to the weighting of the cochains (the initial weights placed on vertices based on their neighbors). This would not prevent implemenation, but would probably make it take some committed time to getting to work.
  - Defects to detect. This repo contains code to generate two defects, holes and dents. As-is, SCNNs would be overengineered to resolve questions related to detecting these on a dataset as small as the subset we use here, as persistence is already sufficient to the task armed with just a MLP-type classifier. However, for more complex datasets with more products and more types of defects, a more robust network may be significantly more successful.
- Persistence added immense signal and classification quality for this task, but could perhaps be faster to preprocess. Barcodes are much easier to train on, however, as the size on disk for the full dataset was only 2MB.
- The data used in this repo is sourced from a collection using real scans, but the defects introduced are synthetic. Real defects are likely not so clean cut, and often 'how' something has failed requires its own specialized classification system. It is likely that, in a production environment, a network similar to this would very successfully detect severe faults, but may become confused by significant variance in parts, data defects, unexpected failure modes, etc.
  - Making a network like this robust against more object types is easy enough, the training set needs to be expanded. If the object type is also known beforehand, this can be provided to the network. Improvements could also be made by simulating different types of failures more robustly.
- To become a production fault detection system, it would be helpful to have a large annotated dataset with various parts and failure modes to derive information from. For holes and dents, persistent homology was the correct tool. However, for more shallow dents, structural instability, or more complicated failure modes, a more sophisticated network/data structure may make sense. Having a large dataset to work with through an approach like this would make it easier to identify what works and what doesn't.
- With a small (or large) dataset of real defective parts, the approach here could remain largely unchanged, though possibly with more classes for defect types.
- As is, I would not ship either classifier. The topological classifier is operating at 96% accuracy on validation, which is good for such a small network, but can likely be improved and tailor-fit to a customer's use case. Likely, the accuracy can be brought higher on real data with a more realistic denting process as well. The Conv3D network operates at around 87% accuracy, which is not at all ideal for this task. It is likely increasing the voxel resolution would improve performance, but likely not to the point it matches the persistent network.
