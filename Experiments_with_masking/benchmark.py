import os
import numpy as np
import pickle
import itertools
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
import yaml
from pathlib import Path
from joblib import Parallel, delayed, Memory
from baselineClassifiers import GCN, GAT, GraphSAGE, MLP
from baselineEmbedders import generate_embeddings
from datasets import downloadDatasets
import shelve

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

here = Path(__file__).resolve().parent
memory = Memory(location=here / 'checkpoints/cache', compress=True)

def createEmbeddings(config, resultsPath = here / 'checkpoints/embeddings'):
    
    results = {'embedding_dict': {}, 'A': {}, 'execution_times': {}, 'masked_labels': {}, 'ground_truth_labels': {}, 'labels_to_be_masked': {}}
    # results = shelve.open(str(resultsPath.resolve()), 'c')

    # try:
    #     with open(resultsPath, 'rb') as file:
    #         results = pickle.load(file)
    # except FileNotFoundError:
    #     results = {'embedding_dict': {}, 'execution_times': {}}
    
    datasets = config['DATASETS']
    models = config['MODELS']['EMBEDDING']
    # benchSeed = rng.integers(1000000, size=(len(datasets), len(models)))

    embedTasks = []
    for dataset, model in itertools.product(datasets, models):
        for mech, rate in itertools.product(config['MISSINGNESS']['mechs'], config['MISSINGNESS']['rates']):
            embedTasks.extend([delayed(generate_embeddings)(dataset, model, config['EMBEDDING_HYPERPARAMETERS'], rate, mech, seed=rng.integers(1000000)) for _ in range(config['NREPS'])])

    # embedTasks, embedData = createEmbeddingTasks(datasets, models, results['embedding_dict'], config['NREPS'], config['HYPERPARAMETERS'], rng)
    missingScenarios = len(config['MISSINGNESS']['mechs']) * len(config['MISSINGNESS']['rates'])
    if embedTasks:
        embedResults = Parallel(n_jobs=config['NJOBS'])(embedTasks)
        for i, (dataset, model) in enumerate(itertools.product(datasets, models)):
            for k, (mech, rate) in enumerate(itertools.product(config['MISSINGNESS']['mechs'], config['MISSINGNESS']['rates'])):
                for j in range(config['NREPS']):
                    if dataset in results['embedding_dict']:
                        if model in results['embedding_dict'][dataset]:
                            if mech in results['embedding_dict'][dataset][model]:
                                if rate in results['embedding_dict'][dataset][model][mech]:
                                    results['embedding_dict'][dataset][model][mech][rate].append(embedResults[config['NREPS'] * (i * missingScenarios + k) + j][0])
                                    results['A'][dataset][model][mech][rate].append(embedResults[config['NREPS'] * (i * missingScenarios + k) + j][1])
                                    results['execution_times'][dataset][model][mech][rate].append(embedResults[config['NREPS'] * (i * missingScenarios + k) + j][2])
                                    results['masked_labels'][dataset][model][mech][rate].append(embedResults[config['NREPS'] * (i * missingScenarios + k) + j][3])
                                    results['ground_truth_labels'][dataset][model][mech][rate].append(embedResults[config['NREPS'] * (i * missingScenarios + k) + j][4])
                                    results['labels_to_be_masked'][dataset][model][mech][rate].append(embedResults[config['NREPS'] * (i * missingScenarios + k) + j][5])
                                else:
                                    results['embedding_dict'][dataset][model][mech][rate] = [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][0]]
                                    results['A'][dataset][model][mech][rate] = [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][1]]
                                    results['execution_times'][dataset][model][mech][rate] = [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][2]]
                                    results['masked_labels'][dataset][model][mech][rate] = [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][3]]
                                    results['ground_truth_labels'][dataset][model][mech][rate] = [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][4]]
                                    results['labels_to_be_masked'][dataset][model][mech][rate] = [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][5]]
                            else:
                                results['embedding_dict'][dataset][model][mech] = {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][0]]}
                                results['A'][dataset][model][mech] = {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][1]]}
                                results['execution_times'][dataset][model][mech] = {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][2]]}
                                results['masked_labels'][dataset][model][mech] = {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][3]]}
                                results['ground_truth_labels'][dataset][model][mech] = {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][4]]}
                                results['labels_to_be_masked'][dataset][model][mech] = {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][5]]}
                        else:
                            results['embedding_dict'][dataset][model] = {mech: {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][0]]}}
                            results['A'][dataset][model] = {mech: {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][1]]}}
                            results['execution_times'][dataset][model] = {mech: {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][2]]}}
                            results['masked_labels'][dataset][model] = {mech: {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][3]]}}
                            results['ground_truth_labels'][dataset][model] = {mech: {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][4]]}}
                            results['labels_to_be_masked'][dataset][model] = {mech: {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][5]]}}
                    else:
                        results['embedding_dict'][dataset] = {model: {mech: {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][0]]}}}
                        results['A'][dataset] = {model: {mech: {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][1]]}}}
                        results['execution_times'][dataset] = {model: {mech: {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][2]]}}}
                        results['masked_labels'][dataset] = {model: {mech: {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][3]]}}}
                        results['ground_truth_labels'][dataset] = {model: {mech: {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][4]]}}}
                        results['labels_to_be_masked'][dataset] = {model: {mech: {rate: [embedResults[config['NREPS'] * (i * missingScenarios + k) + j][5]]}}}
        print(results['embedding_dict'].keys())
        for dataset, model, mech, rate in itertools.product(datasets, models, config['MISSINGNESS']['mechs'], config['MISSINGNESS']['rates']):
            results['embedding_dict'][dataset][model][mech][rate] = np.stack(results['embedding_dict'][dataset][model][mech][rate])
            results['A'][dataset][model][mech][rate] = np.stack(results['A'][dataset][model][mech][rate])
            results['execution_times'][dataset][model][mech][rate] = np.stack(results['execution_times'][dataset][model][mech][rate])
            results['masked_labels'][dataset][model][mech][rate] = np.stack(results['masked_labels'][dataset][model][mech][rate])
            results['ground_truth_labels'][dataset][model][mech][rate] = np.stack(results['ground_truth_labels'][dataset][model][mech][rate])
    
    # with open(resultsPath, 'wb') as file:
    #     pickle.dump(results, file)

    if not resultsPath.exists():
        resultsPath.mkdir(parents=True, exist_ok=True)

    embeddingDict = shelve.open(str(resultsPath.resolve() / 'embedding_dict'), 'c')
    A = shelve.open(str(resultsPath.resolve() / 'A'), 'c')
    executionTimes = shelve.open(str(resultsPath.resolve() / 'execution_times'), 'c')
    maskedLabels = shelve.open(str(resultsPath.resolve() / 'masked_labels'), 'c')
    groundTruthLabels = shelve.open(str(resultsPath.resolve() / 'ground_truth_labels'), 'c')
    labelsToBeMasked = shelve.open(str(resultsPath.resolve() / 'labels_to_be_masked'), 'c')

    for dataset in datasets:
        embeddingDict[dataset] = results['embedding_dict'][dataset]
        print('Embeddings done')
        A[dataset] = results['A'][dataset]
        print('Adjacencies done')
        executionTimes[dataset] = results['execution_times'][dataset]
        print('Execution times done')
        maskedLabels[dataset] = results['masked_labels'][dataset]
        print('Masked labels done')
        groundTruthLabels[dataset] = results['ground_truth_labels'][dataset]
        print('Ground truth labels done')
        labelsToBeMasked[dataset] = results['labels_to_be_masked'][dataset]

    embeddingDict.close()
    A.close()
    executionTimes.close()
    maskedLabels.close()
    groundTruthLabels.close()
    labelsToBeMasked.close()

def trainClassifier(model, embedding, adjacency, labels, nLabels: int, hyperparams: dict, seed: int):

    if model == 'GCN':
        model = GCN(nLabels, seed=seed)
    elif model == 'GAT':
        model = GAT(nLabels, num_heads=hyperparams['num_heads'], seed=seed)
    elif model == 'GraphSAGE':
        model = GraphSAGE(nLabels, hidden_dim=hyperparams['hidden_dim'], aggregator=hyperparams['aggregator'], seed=seed)
    elif model == 'MLP':
        model = MLP(nLabels, hidden_dim=hyperparams['hidden_dim'], depth=hyperparams['depth'], seed=seed)
    else:
        raise ValueError(f"Model {model} not found")

    optimizer = Adam(learning_rate=hyperparams['learning_rate'])
    loss_fn = CategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[CategoricalAccuracy()]
    )

    epochs = hyperparams['epochs']
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions, intermediate_embeddings = model([embedding, adjacency])
            supervised_loss = loss_fn(labels, predictions)

        gradients = tape.gradient(supervised_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if epoch % hyperparams['printEvery'] == 0:
            accuracy = CategoricalAccuracy()(labels, predictions)
            print(f"Epoch {epoch + 1}, Loss: {supervised_loss.numpy()}, Accuracy: {accuracy.numpy()}")

    return model

def evaluateModel(true_labels, predicted_labels):

    # print(true_labels.shape, predicted_labels.shape)
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    cm = confusion_matrix(true_labels, predicted_labels)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'cm': cm
    }

@memory.cache
def trainAndEvaluate(classifier, train_mask, embedding, adjacency, labels, labels_to_be_masked, hyperparams: dict, seed: int):

    X_train = embedding[train_mask]
    Y_train = labels[train_mask]
    Y_train = tf.cast(Y_train, dtype='int32')

    A_train = adjacency[train_mask, :][:, train_mask]
    A_coo = A_train.tocoo()
    indices = np.column_stack((A_coo.row, A_coo.col))
    values = A_coo.data
    shape = A_coo.shape
    A_train_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    A_train_tensor = tf.sparse.reorder(A_train_tensor)
    nLabels = labels.shape[1]

    model = trainClassifier(classifier, X_train, A_train_tensor, Y_train, nLabels, hyperparams, seed)

    X_full = embedding
    A_full = adjacency

    A_full_coo = A_full.tocoo()
    indices_full = np.column_stack((A_full_coo.row, A_full_coo.col))
    values_full = A_full_coo.data
    shape_full = A_full_coo.shape
    A_full_tensor = tf.sparse.SparseTensor(indices=indices_full, values=values_full, dense_shape=shape_full)
    A_full_tensor = tf.sparse.reorder(A_full_tensor)
    
    predictions, emb = model([X_full, A_full_tensor])
    predicted_labels = tf.argmax(predictions, axis=1).numpy()
    predicted_labels_masked = predicted_labels[labels_to_be_masked]
    true_labels_masked = np.argmax(labels, axis=1)[labels_to_be_masked]

    results = evaluateModel(true_labels_masked, predicted_labels_masked)

    return results

def classify(embeddings, A, masked_labels, ground_truth_labels, labels_to_be_masked, hyperparams: dict, classifiers: list[str], rng: np.random.Generator):

    classificationTasks = []
    for dataset in embeddings.keys():
        for model in embeddings[dataset].keys():
            for mech in embeddings[dataset][model].keys():
                for rate in embeddings[dataset][model][mech].keys():
                    train_mask = masked_labels[dataset][model][mech][rate] != -1
                    
                    for classifier, idx in itertools.product(classifiers, range(embeddings[dataset][model][mech][rate].shape[0])):
                        # classificationTasks.append(delayed(trainClassifier)(
                        #     classifier,
                        #     X_train[idx],
                        #     tf.sparse.splice(A_train_tensor, (idx, 0, 0), (1, shape[1], shape[2])),
                        #     Y_train[idx],
                        #     nLabels,
                        #     hyperparams['CLASSIFIER_HYPERPARAMETERS'],
                        #     seed=rng.integers(1000000)
                        # ))
                        classificationTasks.append(delayed(trainAndEvaluate)(
                            classifier,
                            train_mask[idx],
                            embeddings[dataset][model][mech][rate][idx],
                            A[dataset][model][mech][rate][idx],
                            ground_truth_labels[dataset][model][mech][rate][idx],
                            labels_to_be_masked[dataset][model][mech][rate][idx],
                            hyperparams,
                            seed=rng.integers(1000000)
                        ))

    classificationResults = Parallel(n_jobs=config['NJOBS'])(classificationTasks)
    classificationOutputs = {}
    for dsIdx, dataset in enumerate(embeddings.keys()):
        numModels = len(embeddings[dataset].keys())
        classificationOutputs[dataset] = {}
        for modelIdx, model in enumerate(embeddings[dataset].keys()):
            numMechs = len(embeddings[dataset][model].keys())
            classificationOutputs[dataset][model] = {}
            for mechIdx, mech in enumerate(embeddings[dataset][model].keys()):
                numRates = len(embeddings[dataset][model][mech].keys())
                classificationOutputs[dataset][model][mech] = {}
                for rateIdx, rate in enumerate(embeddings[dataset][model][mech].keys()):
                    classificationOutputs[dataset][model][mech][rate] = {}
                    for clfIdx, classifier in enumerate(classifiers):
                        classificationOutputs[dataset][model][mech][rate][classifier] = np.stack([
                            classificationResults[((((dsIdx * numModels + modelIdx) * numMechs + mechIdx) * numRates + rateIdx) * len(classifiers) + clfIdx) * embeddings[dataset][model][mech][rate].shape[0] + i] for i in range(embeddings[dataset][model][mech][rate].shape[0])
                        ])

    return classificationOutputs


if __name__ == "__main__":
    
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    rng = np.random.default_rng(config['SEED'])

    downloadDatasets(config['DATASETS'], here / 'datasets')
    
    embeddingsPath = here / 'checkpoints/embeddings'
    if not embeddingsPath.exists():
        createEmbeddings(config, embeddingsPath)
    
    # with open(embeddingsPath, 'rb') as file:
    #     embeddings = pickle.load(file)

    embeddingDict = shelve.open(str(embeddingsPath.resolve() / 'embedding_dict'), 'r')
    A = shelve.open(str(embeddingsPath.resolve() / 'A'), 'r')
    maskedLabels = shelve.open(str(embeddingsPath.resolve() / 'masked_labels'), 'r')
    groundTruthLabels = shelve.open(str(embeddingsPath.resolve() / 'ground_truth_labels'), 'r')
    labelsToBeMasked = shelve.open(str(embeddingsPath.resolve() / 'labels_to_be_masked'), 'r')

    
    classificationOutputs = classify(
        embeddingDict,
        A,
        maskedLabels,
        groundTruthLabels,
        labelsToBeMasked,
        config['CLASSIFIER_HYPERPARAMETERS'],
        config['MODELS']['CLASSIFIER'],
        rng
    )

    embeddingDict.close()
    A.close()
    maskedLabels.close()
    groundTruthLabels.close()
    labelsToBeMasked.close()

    classificationResultsPath = here / 'checkpoints/classificationResults.pkl'
    with open(classificationResultsPath, 'wb') as file:
        pickle.dump(classificationOutputs, file)