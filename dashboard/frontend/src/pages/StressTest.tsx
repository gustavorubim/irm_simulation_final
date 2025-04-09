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
  SelectChangeEvent,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Alert
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  Cell
} from 'recharts';
import { stressTestService, modelsService, chartsService } from '../services/api';

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

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const StressTest: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(true);
  const [running, setRunning] = useState<boolean>(false);
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>([]);
  const [eveModels, setEveModels] = useState<string[]>([]);
  const [evsModels, setEvsModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [stressTestResults, setStressTestResults] = useState<any>(null);
  const [stressTestChart, setStressTestChart] = useState<string>('');
  const [alert, setAlert] = useState<{type: 'success' | 'error' | 'info', message: string} | null>(null);

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        
        // Fetch stress test scenarios
        const scenariosResponse = await stressTestService.getScenarios();
        if (scenariosResponse.status === 'success') {
          setScenarios(scenariosResponse.scenarios || []);
        }
        
        // Fetch models
        const modelsResponse = await modelsService.getModels();
        setEveModels(modelsResponse.eve_models || []);
        setEvsModels(modelsResponse.evs_models || []);
        
        setLoading(false);
      } catch (error) {
        console.error('Error fetching initial data:', error);
        setAlert({
          type: 'error',
          message: 'Failed to fetch initial data. Please refresh the page.'
        });
        setLoading(false);
      }
    };
    
    fetchInitialData();
  }, []);

  // Handle scenario selection changes
  const handleScenarioSelectionChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    setSelectedScenarios(typeof value === 'string' ? value.split(',') : value);
  };

  // Handle model selection changes
  const handleModelSelectionChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    setSelectedModels(typeof value === 'string' ? value.split(',') : value);
  };

  // Run stress test
  const runStressTest = async () => {
    if (selectedModels.length === 0) {
      setAlert({
        type: 'error',
        message: 'Please select at least one model to run the stress test.'
      });
      return;
    }
    
    if (selectedScenarios.length === 0) {
      setAlert({
        type: 'error',
        message: 'Please select at least one scenario to run the stress test.'
      });
      return;
    }
    
    setRunning(true);
    setAlert(null);
    
    try {
      const response = await stressTestService.runStressTest({
        model_names: selectedModels,
        scenarios: selectedScenarios
      });
      
      if (response.status === 'success') {
        setStressTestResults(response.results);
        setAlert({
          type: 'success',
          message: 'Stress test completed successfully!'
        });
        
        // Fetch stress test chart
        const chartResponse = await chartsService.getStressTest();
        if (chartResponse.status === 'success') {
          setStressTestChart(chartResponse.chart);
        }
      } else {
        setAlert({
          type: 'error',
          message: 'Stress test failed. Please check the configuration and try again.'
        });
      }
    } catch (error) {
      console.error('Error running stress test:', error);
      setAlert({
        type: 'error',
        message: 'An error occurred while running the stress test.'
      });
    } finally {
      setRunning(false);
    }
  };

  // Prepare data for charts
  const prepareChartData = () => {
    if (!stressTestResults) return [];
    
    const chartData: any[] = [];
    
    // For each scenario
    Object.entries(stressTestResults).forEach(([scenario, modelResults]: [string, any]) => {
      const dataPoint: any = { scenario };
      
      // Add each model's result
      Object.entries(modelResults).forEach(([model, value]: [string, any]) => {
        dataPoint[model] = value;
      });
      
      chartData.push(dataPoint);
    });
    
    return chartData;
  };

  const chartData = prepareChartData();

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Stress Test
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
            <CardHeader title="Stress Test Configuration" />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel id="scenario-selection-label">Select Scenarios</InputLabel>
                    <Select
                      labelId="scenario-selection-label"
                      id="scenario-selection"
                      multiple
                      value={selectedScenarios}
                      onChange={handleScenarioSelectionChange}
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {selected.map((value) => (
                            <Chip key={value} label={value} />
                          ))}
                        </Box>
                      )}
                    >
                      {scenarios.map((scenario) => (
                        <MenuItem key={scenario} value={scenario}>
                          {scenario}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                
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
                
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="primary"
                    fullWidth
                    onClick={runStressTest}
                    disabled={running || selectedModels.length === 0 || selectedScenarios.length === 0}
                    sx={{ mt: 2 }}
                  >
                    {running ? <CircularProgress size={24} /> : 'Run Stress Test'}
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* Scenario Documentation */}
        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardHeader title="Scenario Documentation" />
            <CardContent>
              <Typography variant="body1" paragraph>
                Stress tests evaluate the resilience of models under various adverse scenarios. 
                The following scenarios are available for testing:
              </Typography>
              
              <TableContainer component={Paper} elevation={0}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Scenario</TableCell>
                      <TableCell>Description</TableCell>
                      <TableCell>Severity</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>parallel_up_200bp</TableCell>
                      <TableCell>Parallel upward shift of 200 basis points across all rate tenors</TableCell>
                      <TableCell><Chip label="High" color="error" size="small" /></TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>parallel_down_200bp</TableCell>
                      <TableCell>Parallel downward shift of 200 basis points across all rate tenors</TableCell>
                      <TableCell><Chip label="High" color="error" size="small" /></TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>steepener</TableCell>
                      <TableCell>Short rates down, long rates up (steeper yield curve)</TableCell>
                      <TableCell><Chip label="Medium" color="warning" size="small" /></TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>flattener</TableCell>
                      <TableCell>Short rates up, long rates down (flatter yield curve)</TableCell>
                      <TableCell><Chip label="Medium" color="warning" size="small" /></TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>short_up</TableCell>
                      <TableCell>Short rates up (up to 2Y), others unchanged</TableCell>
                      <TableCell><Chip label="Medium" color="warning" size="small" /></TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>short_down</TableCell>
                      <TableCell>Short rates down (up to 2Y), others unchanged</TableCell>
                      <TableCell><Chip label="Medium" color="warning" size="small" /></TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>long_up</TableCell>
                      <TableCell>Long rates up (5Y and above), others unchanged</TableCell>
                      <TableCell><Chip label="Medium" color="warning" size="small" /></TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>long_down</TableCell>
                      <TableCell>Long rates down (5Y and above), others unchanged</TableCell>
                      <TableCell><Chip label="Medium" color="warning" size="small" /></TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>historical_2008</TableCell>
                      <TableCell>Rate changes similar to 2008 financial crisis</TableCell>
                      <TableCell><Chip label="Extreme" color="error" size="small" /></TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>historical_2020</TableCell>
                      <TableCell>Rate changes similar to 2020 COVID-19 crisis</TableCell>
                      <TableCell><Chip label="High" color="error" size="small" /></TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* Results Section - only show if stress test is complete */}
        {stressTestResults && (
          <>
            {/* Stress Test Chart */}
            <Grid item xs={12}>
              <StyledCard>
                <CardHeader 
                  title="Stress Test Results" 
                  subheader="Impact of different scenarios on model values"
                />
                <CardContent sx={{ flexGrow: 1 }}>
                  {stressTestChart ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                      <img src={stressTestChart} alt="Stress Test Results" style={{ maxWidth: '100%', maxHeight: 500 }} />
                    </Box>
                  ) : (
                    <ResponsiveContainer width="100%" height={500}>
                      <BarChart
                        data={chartData}
                        margin={{
                          top: 20,
                          right: 30,
                          left: 20,
                          bottom: 100,
                        }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="scenario" 
                          angle={-45} 
                          textAnchor="end"
                          height={80}
                          interval={0}
                        />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        {selectedModels.map((model, index) => (
                          <Bar 
                            key={model} 
                            dataKey={model} 
                            fill={COLORS[index % COLORS.length]} 
                          />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </StyledCard>
            </Grid>
            
            {/* Detailed Results Table */}
            <Grid item xs={12}>
              <StyledCard>
                <CardHeader 
                  title="Detailed Results" 
                  subheader="Numerical results for each model and scenario"
                />
                <CardContent>
                  <TableContainer component={Paper} elevation={0}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Scenario</TableCell>
                          {selectedModels.map((model) => (
                            <TableCell key={model}>{model}</TableCell>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(stressTestResults).map(([scenario, modelResults]: [string, any]) => (
                          <TableRow key={scenario}>
                            <TableCell>{scenario}</TableCell>
                            {selectedModels.map((model) => (
                              <TableCell key={model}>
                                {modelResults[model] !== undefined ? modelResults[model].toFixed(2) : 'N/A'}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </StyledCard>
            </Grid>
            
            {/* Heatmap View */}
            <Grid item xs={12}>
              <StyledCard>
                <CardHeader 
                  title="Sensitivity Heatmap" 
                  subheader="Visual representation of model sensitivity to scenarios"
                />
                <CardContent>
                  <TableContainer component={Paper} elevation={0}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Scenario</TableCell>
                          {selectedModels.map((model) => (
                            <TableCell key={model}>{model}</TableCell>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(stressTestResults).map(([scenario, modelResults]: [string, any]) => {
                          // Find min and max values for color scaling
                          const values = selectedModels.map(model => modelResults[model] || 0);
                          const minValue = Math.min(...values);
                          const maxValue = Math.max(...values);
                          const range = maxValue - minValue;
                          
                          return (
                            <TableRow key={scenario}>
                              <TableCell>{scenario}</TableCell>
                              {selectedModels.map((model) => {
                                const value = modelResults[model];
                                
                                // Calculate color based on value
                                let backgroundColor = '#ffffff';
                                if (value !== undefined) {
                                  if (value < 0) {
                                    // Red for negative values, darker for more negative
                                    const intensity = Math.min(1, Math.abs(value) / (Math.abs(minValue) || 1));
                                    backgroundColor = `rgba(255, 0, 0, ${intensity * 0.7})`;
                                  } else {
                                    // Green for positive values, darker for more positive
                                    const intensity = Math.min(1, value / (maxValue || 1));
                                    backgroundColor = `rgba(0, 128, 0, ${intensity * 0.7})`;
                                  }
                                }
                                
                                return (
                                  <TableCell 
                                    key={model} 
                                    sx={{ 
                                      backgroundColor,
                                      color: backgroundColor !== '#ffffff' ? 'white' : 'inherit'
                                    }}
                                  >
                                    {value !== undefined ? value.toFixed(2) : 'N/A'}
                                  </TableCell>
                                );
                              })}
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </StyledCard>
            </Grid>
          </>
        )}
      </Grid>
    </Box>
  );
};

export default StressTest;
