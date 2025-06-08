import os
import torch
from models.ordered import OrderedNetwork
from models.group_lasso import GroupLassoNetwork
from models.post_train_pruning import PostTrainPruningNetwork
from utils.utils import train, test, adjust_weights

models = [OrderedNetwork(0.5, 0.1), GroupLassoNetwork(1e-4), PostTrainPruningNetwork()]
epochs = 10
optimizer = torch.optim.Adam
criterion = torch.nn.CrossEntropyLoss()
pruning_steps = 32 # Number of steps in model pruning
adjusting_steps = 10 # Number of steps to adjust weights after pruning

for model in models:
    # Train the full model
    print(f'Training {model.name}...')
    train(model, epochs, optimizer(model.parameters()), criterion)
    accuracy = test(model)
    print(f'Full model accuracy: {accuracy * 100:.2f}%')

    # Test various pruning amounts
    results = [0.1] # Since there are 10 classes, 0% model size has 10% accuracy
    for i in range(pruning_steps):
        amount = (i + 1) / pruning_steps
        pruned = model.prune(amount)
        adjust_weights(pruned, adjusting_steps, optimizer(model.parameters()), criterion)
        
        accuracy = test(pruned)
        results.append(accuracy)
        print(f'Model size: {amount * 100:.2f}%, Accuracy: {accuracy * 100:.2f}%')
    
    print()

    # Save the results

    if not os.path.exists('output'):
        os.makedirs('output')

    with open(f'output/{model.name}.txt', 'w') as f:
        for i, accuracy in enumerate(results):
            f.write(f'{i / pruning_steps * 100:.2f}%: {accuracy * 100:.2f}%\n')
