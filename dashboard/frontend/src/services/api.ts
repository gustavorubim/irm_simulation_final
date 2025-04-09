import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

// API service for models
export const modelsService = {
  getModels: async () => {
    try {
      const response = await axios.get(`${API_URL}/models`);
      return response.data;
    } catch (error) {
      console.error('Error fetching models:', error);
      throw error;
    }
  },
  
  getModelDetails: async () => {
    try {
      const response = await axios.get(`${API_URL}/model-details`);
      return response.data;
    } catch (error) {
      console.error('Error fetching model details:', error);
      throw error;
    }
  }
};

// API service for simulation
export const simulationService = {
  getConfig: async () => {
    try {
      const response = await axios.get(`${API_URL}/simulation/config`);
      return response.data;
    } catch (error) {
      console.error('Error fetching simulation config:', error);
      throw error;
    }
  },
  
  updateConfig: async (config: any) => {
    try {
      const response = await axios.post(`${API_URL}/simulation/config`, config);
      return response.data;
    } catch (error) {
      console.error('Error updating simulation config:', error);
      throw error;
    }
  },
  
  runSimulation: async (params: any) => {
    try {
      const response = await axios.post(`${API_URL}/simulation/run`, params);
      return response.data;
    } catch (error) {
      console.error('Error running simulation:', error);
      throw error;
    }
  },
  
  getResults: async () => {
    try {
      const response = await axios.get(`${API_URL}/simulation/results`);
      return response.data;
    } catch (error) {
      console.error('Error fetching simulation results:', error);
      throw error;
    }
  },
  
  getTimeseries: async (model: string, scenarios: number = 10) => {
    try {
      const response = await axios.get(`${API_URL}/simulation/timeseries?model=${model}&scenarios=${scenarios}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching simulation timeseries:', error);
      throw error;
    }
  }
};

// API service for rates
export const ratesService = {
  getTimeseries: async (tenor: string, scenarios: number = 10) => {
    try {
      const response = await axios.get(`${API_URL}/rates/timeseries?tenor=${tenor}&scenarios=${scenarios}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching rates timeseries:', error);
      throw error;
    }
  }
};

// API service for stress tests
export const stressTestService = {
  getScenarios: async () => {
    try {
      const response = await axios.get(`${API_URL}/stress-test/scenarios`);
      return response.data;
    } catch (error) {
      console.error('Error fetching stress test scenarios:', error);
      throw error;
    }
  },
  
  runStressTest: async (params: any) => {
    try {
      const response = await axios.post(`${API_URL}/stress-test/run`, params);
      return response.data;
    } catch (error) {
      console.error('Error running stress test:', error);
      throw error;
    }
  },
  
  getResults: async () => {
    try {
      const response = await axios.get(`${API_URL}/stress-test/results`);
      return response.data;
    } catch (error) {
      console.error('Error fetching stress test results:', error);
      throw error;
    }
  }
};

// API service for metrics
export const metricsService = {
  getMetrics: async () => {
    try {
      const response = await axios.get(`${API_URL}/metrics`);
      return response.data;
    } catch (error) {
      console.error('Error fetching metrics:', error);
      throw error;
    }
  }
};

// API service for portfolio
export const portfolioService = {
  optimizePortfolio: async (params: any) => {
    try {
      const response = await axios.post(`${API_URL}/portfolio/optimize`, params);
      return response.data;
    } catch (error) {
      console.error('Error optimizing portfolio:', error);
      throw error;
    }
  },
  
  analyzePortfolio: async (params: any) => {
    try {
      const response = await axios.post(`${API_URL}/portfolio/analyze`, params);
      return response.data;
    } catch (error) {
      console.error('Error analyzing portfolio:', error);
      throw error;
    }
  }
};

// API service for charts
export const chartsService = {
  getHistogram: async (model: string, bins: number = 50) => {
    try {
      const response = await axios.get(`${API_URL}/charts/histogram?model=${model}&bins=${bins}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching histogram chart:', error);
      throw error;
    }
  },
  
  getCorrelation: async () => {
    try {
      const response = await axios.get(`${API_URL}/charts/correlation`);
      return response.data;
    } catch (error) {
      console.error('Error fetching correlation chart:', error);
      throw error;
    }
  },
  
  getSensitivity: async () => {
    try {
      const response = await axios.get(`${API_URL}/charts/sensitivity`);
      return response.data;
    } catch (error) {
      console.error('Error fetching sensitivity chart:', error);
      throw error;
    }
  },
  
  getStressTest: async () => {
    try {
      const response = await axios.get(`${API_URL}/charts/stress-test`);
      return response.data;
    } catch (error) {
      console.error('Error fetching stress test chart:', error);
      throw error;
    }
  }
};
