import { COLORS, createColorExample } from './colorData.js';

const TRAINING_EXAMPLES_PER_COLOR = 40;
const TRAINING_OPTIONS = { epochs: 40, batchSize: 12 };

// Build labeled RGB examples: TRAINING_EXAMPLES_PER_COLOR for each color,
// as objects shaped { red, green, blue, label }.
function createTrainingExamples() {
    return COLORS.flatMap((color) =>
        Array.from({ length: TRAINING_EXAMPLES_PER_COLOR }, () =>
            createColorExample(color),
        ),
    );
}
console.log(createTrainingExamples());
export default class ML {
    constructor() {
        this.model = null;
    }
    // Create labeled RGB examples, train the ml5 model, and report progress.
    async trainModel(updateProgress) {
        if (typeof ml5 === 'undefined') {
            throw new Error(
                'ml5.js is not loaded. Check your internet connection and reload the page.',
            );
        }

        const report = (message) => updateProgress?.(message);
        const totalEpochs = TRAINING_OPTIONS.epochs;

        // Clear any old model so a failed retrain cannot leave a stale one.
        this.model = null;

        report('Creating training data...');
        const model = ml5.neuralNetwork({ task: 'classification' });
        for (const { red, green, blue, label } of createTrainingExamples()) {
            model.addData([red, green, blue], [label]);
        }
        model.normalizeData();

        report('Training model...');
        await new Promise((resolve, reject) => {
            const whileTraining = (epoch) => {
                const done = Math.min(epoch + 1, totalEpochs);
                report(`Training model... epoch ${done} of ${totalEpochs}`);
            };
            // Resolve only when ml5 says training is finished. If train()
            // throws, or returns a promise that rejects, the error is passed
            // on to the caller.
            const pending = model.train(
                { ...TRAINING_OPTIONS },
                whileTraining,
                resolve,
            );
            pending?.catch?.(reject);
        });

        this.model = model;
    }

    // Classify an RGB color with the trained model and return its label.
    async classify(color) {}


}
