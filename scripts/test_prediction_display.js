#!/usr/bin/env node
/**
 * Test script to verify prediction display updates correctly
 * 
 * This simulates different classification prediction responses
 * to ensure the dashboard displays the correct flare class and probability
 */

const testCases = [
  {
    name: "High M-class probability",
    response: {
      success: true,
      result: {
        predictedClass: "M",
        probabilities: {
          "None": 0.1,
          "C": 0.2,
          "M": 0.65,
          "X": 0.05
        }
      }
    },
    expected: {
      level: "M",
      label: "Strong",
      prob: 0.65
    }
  },
  {
    name: "Low C-class probability",
    response: {
      success: true,
      result: {
        predictedClass: "C",
        probabilities: {
          "None": 0.7,
          "C": 0.25,
          "M": 0.04,
          "X": 0.01
        }
      }
    },
    expected: {
      level: "C",
      label: "Minor",
      prob: 0.25
    }
  },
  {
    name: "X-class extreme",
    response: {
      success: true,
      result: {
        predictedClass: "X",
        probabilities: {
          "None": 0.05,
          "C": 0.1,
          "M": 0.2,
          "X": 0.65
        }
      }
    },
    expected: {
      level: "X",
      label: "Extreme",
      prob: 0.65
    }
  },
  {
    name: "No flare expected",
    response: {
      success: true,
      result: {
        predictedClass: "None",
        probabilities: {
          "None": 0.95,
          "C": 0.04,
          "M": 0.01,
          "X": 0.0
        }
      }
    },
    expected: {
      level: "None",
      label: "None",
      prob: 0.95
    }
  }
];

function simulatePredictionLogic(response) {
  const classProbs = response.result.probabilities || {};
  const predictedClass = response.result.predictedClass || "None";
  
  // Find the highest probability flare class
  // Prefer flare classes (C, M, X) over "None" if they have significant probability
  let maxProb = 0;
  let maxClass = "None";
  let maxProbValue = classProbs["None"] || 0;
  const noneProb = classProbs["None"] || 0;
  
  // Check C, M, X classes (in order of severity)
  const flareClasses = ["C", "M", "X"];
  for (const flareClass of flareClasses) {
    const prob = classProbs[flareClass] || 0;
    if (prob > maxProb) {
      maxProb = prob;
      maxClass = flareClass;
      maxProbValue = prob;
    }
  }
  
  // Only use "None" if no flare class has significant probability (>5%)
  // This ensures we show flare predictions even if "None" is higher
  if (maxProb < 0.05) {
    maxClass = "None";
    maxProbValue = noneProb;
  }
  
  // Determine NOAA-style level
  let predictionLevel;
  if (maxClass === "X") {
    predictionLevel = { 
      level: "X", 
      label: "Extreme", 
      color: "#9333ea", 
      prob: maxProbValue,
      class: maxClass
    };
  } else if (maxClass === "M") {
    if (maxProbValue >= 0.50) {
      predictionLevel = { level: "M", label: "Strong", color: "#f97316", prob: maxProbValue, class: maxClass };
    } else if (maxProbValue >= 0.25) {
      predictionLevel = { level: "M", label: "Moderate", color: "#eab308", prob: maxProbValue, class: maxClass };
    } else {
      predictionLevel = { level: "M", label: "Minor", color: "#84cc16", prob: maxProbValue, class: maxClass };
    }
  } else if (maxClass === "C") {
    if (maxProbValue >= 0.50) {
      predictionLevel = { level: "C", label: "Moderate", color: "#eab308", prob: maxProbValue, class: maxClass };
    } else {
      predictionLevel = { level: "C", label: "Minor", color: "#84cc16", prob: maxProbValue, class: maxClass };
    }
  } else {
    predictionLevel = { level: "None", label: "None", color: "#6b7280", prob: maxProbValue, class: "None" };
  }
  
  return predictionLevel;
}

console.log("Testing prediction display logic...\n");

let passed = 0;
let failed = 0;

for (const testCase of testCases) {
  const result = simulatePredictionLogic(testCase.response);
  const levelMatch = result.level === testCase.expected.level;
  const probMatch = Math.abs(result.prob - testCase.expected.prob) < 0.01;
  
  if (levelMatch && probMatch) {
    console.log(`✅ ${testCase.name}`);
    console.log(`   Expected: ${testCase.expected.level} (${testCase.expected.prob})`);
    console.log(`   Got: ${result.level} (${result.prob.toFixed(2)})`);
    passed++;
  } else {
    console.log(`❌ ${testCase.name}`);
    console.log(`   Expected: ${testCase.expected.level} (${testCase.expected.prob})`);
    console.log(`   Got: ${result.level} (${result.prob.toFixed(2)})`);
    failed++;
  }
  console.log();
}

console.log(`\nResults: ${passed} passed, ${failed} failed`);

if (failed === 0) {
  console.log("✅ All tests passed!");
  process.exit(0);
} else {
  console.log("❌ Some tests failed");
  process.exit(1);
}

