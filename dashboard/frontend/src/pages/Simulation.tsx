import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Grid, 
  Paper, 
  Typography, 
  Card, 
  CardContent, 
  CardHeader,
  Button,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Slider,
  Chip,
  Divider,
  Alert,
  SelectChangeEvent,
  FormGroup,
  FormControlLabel,
  Checkbox
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import { simulationService, modelsService, ratesService } from '../services/api';

const StyledCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'transform 0.3s, box-shadow 0.3s',
  '&:hover': {
    transform: 'translateY(-5px)',
    boxShadow: '0 12px 20px rgba(0, 0, 0, 0.3)',
  },
}));

const Simulation: React.FC = () => {
  // State for models
  const [eveModels, setEveModels] = useState<string[]>([]);
  const [evsModels, setEvsModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  
  // State for simulation configuration
  const [config, setConfig] = useState<any>({
    num_scenarios: 1000,
    time_horizon: 3.0,
    time_steps_per_year: 12,
    rate_tenors: [],
    initial_rates: {},
    volatility_params: {},
    mean_reversion_params: {}
  });
  
  // State for simulation results
  const [running, setRunning] = useState<boolean>(false);
  const [simulationComplete, setSimulationComplete] = useState<boolean>(false);
  const [simulationResults, setSimulationResults] = useState<any>(null);
  const [selectedRateTenor, setSelectedRateTenor] = useState<string>('');
  const [rateTimeseriesData, setRateTimeseriesData] = useState<any[]>([]);
  const [selectedModelForChart, setSelectedModelForChart] = useState<string>('');
  const [modelTimeseriesData, setModelTimeseriesData] = useState<any[]>([]);
  
  // State for alerts
  const [alert, setAlert] = useState<{type: 'success' | 'error' | 'info', message: string} | null>(null);

  // Fetch models and configuration on component mount
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        // Fetch models
        const modelsResponse = await modelsService.getModels();
        setEveModels(modelsResponse.eve_models || []);
        setEvsModels(modelsResponse.evs_models || []);
        
        // Fetch simulation configuration
        const configResponse = await simulationService.getConfig();
        setConfig(configResponse);
        
        // Set default selected rate tenor if available
        if (configResponse.rate_tenors && configResponse.rate_tenors.length > 0) {
          setSelectedRateTenor(configResponse.rate_tenors[0]);
        }
        
      } catch (error) {
        console.error('Error fetching initial data:', error);
        setAlert({
          type: 'error',
          message: 'Failed to fetch initial data. Please refresh the page.'
        });
      }
    };
    
    fetchInitialData();
  }, []);

  // Handle model selection changes
  const handleModelSelectionChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    setSelectedModels(typeof value === 'string' ? value.split(',') : value);
  };

  // Handle configuration changes
  const handleConfigChange = (field: string, value: any) => {
    setConfig({
      ...config,
      [field]: value
    });
  };

  // Handle rate tenor selection for chart
  const handleRateTenorChange = (event: SelectChangeEvent) => {
    setSelectedRateTenor(event.target.value);
    fetchRateTimeseries(event.target.value);
  };

  // Handle model selection for chart
  const handleModelForChartChange = (event: SelectChangeEvent) => {
    setSelectedModelForChart(event.target.value);
    fetchModelTimeseries(event.target.value);
  };

  // Fetch rate timeseries data
  const fetchRateTimeseries = async (tenor: string) => {
    if (!tenor) return;
    
    try {
      const response = await ratesService.getTimeseries(tenor);
      if (response.status === 'success') {
        // Transform data for recharts
        const formattedData = response.steps.map((step: number, index: number) => {
          return {
            step,
            mean: response.statistics.mean[index],
            median: response.statistics.median[index],
            p5: response.statistics.percentile_5[index],
            p95: response.statistics.percentile_95[index],
          };
        });
        setRateTimeseriesData(formattedData);
      }
    } catch (error) {
      console.error('Error fetching rate timeseries:', error);
    }
  };

  // Fetch model timeseries data
  const fetchModelTimeseries = async (model: string) => {
    if (!model) return;
    
    try {
      const response = await simulationService.getTimeseries(model);
      if (response.status === 'success') {
        // Transform data for recharts
        const formattedData = response.steps.map((step: number, index: number) => {
          return {
            step,
            mean: response.statistics.mean[index],
            median: response.statistics.median[index],
            p5: response.statistics.percentile_5[index],
            p95: response.statistics.percentile_95[index],
          };
        });
        setModelTimeseriesData(formattedData);
      }
    } catch (error) {
      console.error('Error fetching model timeseries:', error);
    }
  };

  // Run simulation
  const runSimulation = async () => {
    if (selectedModels.length === 0) {
      setAlert({
        type: 'error',
        message: 'Please select at least one model to run the simulation.'
      });
      return;
    }
    
    setRunning(true);
    setAlert(null);
    
    try {
      const response = await simulationService.runSimulation({
        model_names: selectedModels,
        num_scenarios: config.num_scenarios,
        time_horizon: config.time_horizon,
        config: config
      });
      
      if (response.status === 'success') {
        setSimulationResults(response);
        setSimulationComplete(true);
        setAlert({
          type: 'success',
          message: 'Simulation completed successfully!'
        });
        
        // Set default selected model for chart if available
        if (selectedModels.length > 0) {
          setSelectedModelForChart(selectedModels[0]);
          fetchModelTimeseries(selectedModels[0]);
        }
        
        // Fetch rate timeseries for selected tenor
        if (selectedRateTenor) {
          fetchRateTimeseries(selectedRateTenor);
        }
      } else {
        setAlert({
          type: 'error',
          message: 'Simulation failed. Please check the configuration and try again.'
        });
      }
    } catch (error) {
      console.error('Error running simulation:', error);
      setAlert({
        type: 'error',
        message: 'An error occurred while running the simulation.'
      });
    } finally {
      setRunning(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Simulation
      </Typography>
      
      {alert && (
        <Alert 
          severity={alert.type} 
          sx={{ mb: 3 }}
          onClose={() => setAlert(null)}
        >
          {alert.message}
        </Alert>
      )}
      
      <Grid container spacing={3}>
        {/* Configuration Section */}
        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardHeader title="Simulation Configuration" />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel id="model-selection-label">Select Models</InputLabel>
                    <Select
                      labelId="model-selection-label"
                      id="model-selection"
                      multiple
                      value={selectedModels}
                      onChange={handleModelSelectionChange}
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {selected.map((value) => (
                            <Chip key={value} label={value} />
                          ))}
                        </Box>
                      )}
                    >
                      <MenuItem disabled>
                        <Typography variant="subtitle2">EVE Models</Typography>
                      </MenuItem>
                      {eveModels.map((model) => (
                        <MenuItem key={model} value={model}>
                          {model}
                        </MenuItem>
                      ))}
                      <Divider />
                      <MenuItem disabled>
                        <Typography variant="subtitle2">EVS Models</Typography>
                      </MenuItem>
                      {evsModels.map((model) => (
                        <MenuItem key={model} value={model}>
                          {model}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Number of Scenarios"
                    type="number"
                    value={config.num_scenarios}
                    onChange={(e) => handleConfigChange('num_scenarios', parseInt(e.target.value))}
                    InputProps={{ inputProps: { min: 100, max: 10000 } }}
                  />
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Time Horizon (years)"
                    type="number"
                    value={config.time_horizon}
                    onChange={(e) => handleConfigChange('time_horizon', parseFloat(e.target.value))}
                    InputProps={{ inputProps: { min: 1, max: 10, step: 0.5 } }}
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Typography gutterBottom>
                    Time Steps per Year: {config.time_steps_per_year}
                  </Typography>
                  <Slider
                    value={config.time_steps_per_year}
                    onChange={(_, value) => handleConfigChange('time_steps_per_year', value)}
                    step={4}
                    marks
                    min={4}
                    max={52}
                    valueLabelDisplay="auto"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="primary"
                    fullWidth
                    onClick={runSimulation}
                    disabled={running || selectedModels.length === 0}
                    sx={{ mt: 2 }}
                  >
                    {running ? <CircularProgress size={24} /> : 'Run Simulation'}
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* Advanced Configuration */}
        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardHeader title="Advanced Configuration" />
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Initial Rates
              </Typography>
              <Grid container spacing={2}>
                {config.rate_tenors && config.rate_tenors.map((tenor: string) => (
                  <Grid item xs={12} sm={6} key={tenor}>
                    <TextField
                      fullWidth
                      label={`${tenor} Rate`}
                      type="number"
                      value={config.initial_rates[tenor]}
                      onChange={(e) => {
                        const newInitialRates = { ...config.initial_rates };
                        newInitialRates[tenor] = parseFloat(e.target.value);
                        handleConfigChange('initial_rates', newInitialRates);
                      }}
                      InputProps={{ inputProps: { min: 0, max: 0.2, step: 0.001 } }}
                    />
                  </Grid>
                ))}
              </Grid>
              
              <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}>
                Volatility Parameters
              </Typography>
              <Grid container spacing={2}>
                {config.rate_tenors && config.rate_tenors.map((tenor: string) => (
                  <Grid item xs={12} sm={6} key={tenor}>
                    <TextField
                      fullWidth
                      label={`${tenor} Volatility`}
                      type="number"
                      value={config.volatility_params[tenor]}
                      onChange={(e) => {
                        const newVolatilityParams = { ...config.volatility_params };
                        newVolatilityParams[tenor] = parseFloat(e.target.value);
                        handleConfigChange('volatility_params', newVolatilityParams);
                      }}
                      InputProps={{ inputProps: { min: 0.001, max: 0.05, step: 0.001 } }}
                    />
                  </Grid>
                ))}
              </Grid>
              
              <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}>
                Mean Reversion Parameters
              </Typography>
              <Grid container spacing={2}>
                {config.rate_tenors && config.rate_tenors.map((tenor: string) => (
                  <Grid item xs={12} sm={6} key={tenor}>
                    <TextField
                      fullWidth
                      label={`${tenor} Mean Reversion`}
                      type="number"
                      value={config.mean_reversion_params[tenor]}
                      onChange={(e) => {
                        const newMeanReversionParams = { ...config.mean_reversion_params };
                        newMeanReversionParams[tenor] = parseFloat(e.target.value);
                        handleConfigChange('mean_reversion_params', newMeanReversionParams);
                      }}
                      InputProps={{ inputProps: { min: 0.01, max: 0.5, step: 0.01 } }}
                    />
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* Results Section - only show if simulation is complete */}
        {simulationComplete && (
          <>
            {/* Interest Rate Chart */}
            <Grid item xs={12} md={6}>
              <StyledCard>
                <CardHeader 
                  title="Interest Rate Simulation" 
                  subheader="Projected interest rate paths"
                />
                <CardContent>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel id="rate-tenor-select-label">Rate Tenor</InputLabel>
                    <Select
                      labelId="rate-tenor-select-label"
                      id="rate-tenor-select"
                      value={selectedRateTenor}
                      label="Rate Tenor"
                      onChange={handleRateTenorChange}
                    >
                      {config.rate_tenors && config.rate_tenors.map((tenor: string) => (
                        <MenuItem key={tenor} value={tenor}>{tenor}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={rateTimeseriesData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="step" label={{ value: 'Time Step', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: 'Rate', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Area type="monotone" dataKey="p95" fill="#8884d8" stroke="#8884d8" fillOpacity={0.1} name="95th Percentile" />
                      <Area type="monotone" dataKey="mean" fill="#82ca9d" stroke="#82ca9d" fillOpacity={0.8} name="Mean" />
                      <Area type="monotone" dataKey="median" fill="#ffc658" stroke="#ffc658" fillOpacity={0.5} name="Median" />
                      <Area type="monotone" dataKey="p5" fill="#ff8042" stroke="#ff8042" fillOpacity={0.1} name="5th Percentile" />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </StyledCard>
            </Grid>
            
            {/* Model Results Chart */}
            <Grid item xs={12} md={6}>
              <StyledCard>
                <CardHeader 
                  title="Model Simulation Results" 
                  subheader="Projected model values"
                />
                <CardContent>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel id="model-select-label">Model</InputLabel>
                    <Select
                      labelId="model-select-label"
                      id="model-select"
                      value={selectedModelForChart}
                      label="Model"
                      onChange={handleModelForChartChange}
                    >
                      {selectedModels.map((model) => (
                        <MenuItem key={model} value={model}>{model}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={modelTimeseriesData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="step" label={{ value: 'Time Step', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: 'Value', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Area type="monotone" dataKey="p95" fill="#8884d8" stroke="#8884d8" fillOpacity={0.1} name="95th Percentile" />
                      <Area type="monotone" dataKey="mean" fill="#82ca9d" stroke="#82ca9d" fillOpacity={0.8} name="Mean" />
                      <Area type="monotone" dataKey="median" fill="#ffc658" stroke="#ffc658" fillOpacity={0.5} name="Median" />
                      <Area type="monotone" dataKey="p5" fill="#ff8042" stroke="#ff8042" fillOpacity={0.1} name="5th Percentile" />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </StyledCard>
            </Grid>
            
            {/* Metrics Summary */}
            <Grid item xs={12}>
              <StyledCard>
                <CardHeader 
                  title="Simulation Metrics Summary" 
                  subheader="Key risk metrics from simulation results"
                />
                <CardContent>
                  {simulationResults && simulationResults.metrics && (
                    <Grid container spacing={2}>
                      {selectedModels.map((model) => (
                        <Grid item xs={12} md={4} key={model}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>{model}</Typography>
                            <Divider sx={{ mb: 2 }} />
                            <Grid container spacing={1}>
                              <Grid item xs={6}>
                                <Typography variant="body2" color="text.secondary">Mean:</Typography>
                              </Grid>
                              <Grid item xs={6}>
                                <Typography variant="body2">{simulationResults.metrics[model]?.mean?.toFixed(2) || 'N/A'}</Typography>
                              </Grid>
                              
                              <Grid item xs={6}>
                                <Typography variant="body2" color="text.secondary">Std Dev:</Typography>
                              </Grid>
                              <Grid item xs={6}>
                                <Typography variant="body2">{simulationResults.metrics[model]?.std?.toFixed(2) || 'N/A'}</Typography>
                              </Grid>
                              
                              <Grid item xs={6}>
                                <Typography variant="body2" color="text.secondary">VaR (95%):</Typography>
                              </Grid>
                              <Grid item xs={6}>
                                <Typography variant="body2">{simulationResults.metrics[model]?.var_95?.toFixed(2) || 'N/A'}</Typography>
                              </Grid>
                              
                              <Grid item xs={6}>
                                <Typography variant="body2" color="text.secondary">ES (95%):</Typography>
                              </Grid>
                              <Grid item xs={6}>
                                <Typography variant="body2">{simulationResults.metrics[model]?.es_95?.toFixed(2) || 'N/A'}</Typography>
                              </Grid>
                              
                              <Grid item xs={6}>
                                <Typography variant="body2" color="text.secondary">Sharpe Ratio:</Typography>
                              </Grid>
                              <Grid item xs={6}>
                                <Typography variant="body2">{simulationResults.metrics[model]?.sharpe_ratio?.toFixed(2) || 'N/A'}</Typography>
                              </Grid>
                            </Grid>
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>
                  )}
                </CardContent>
              </StyledCard>
            </Grid>
          </>
        )}
      </Grid>
    </Box>
  );
};

export default Simulation;
