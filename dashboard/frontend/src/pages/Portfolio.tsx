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
  Slider,
  TextField,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { 
  PieChart, 
  Pie, 
  Cell, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';
import { portfolioService, modelsService, simulationService } from '../services/api';

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

const Portfolio: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(true);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [optimizing, setOptimizing] = useState<boolean>(false);
  const [eveModels, setEveModels] = useState<string[]>([]);
  const [evsModels, setEvsModels] = useState<string[]>([]);
  const [selectedComponents, setSelectedComponents] = useState<string[]>([]);
  const [weights, setWeights] = useState<{[key: string]: number}>({});
  const [portfolioResults, setPortfolioResults] = useState<any>(null);
  const [optimizationResults, setOptimizationResults] = useState<any>(null);
  const [objective, setObjective] = useState<string>('min_variance');
  const [alert, setAlert] = useState<{type: 'success' | 'error' | 'info', message: string} | null>(null);
  const [simulationRun, setSimulationRun] = useState<boolean>(false);

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        
        // Fetch models
        const modelsResponse = await modelsService.getModels();
        setEveModels(modelsResponse.eve_models || []);
        setEvsModels(modelsResponse.evs_models || []);
        
        // Check if simulation has been run
        const resultsResponse = await simulationService.getResults();
        if (resultsResponse.status === 'success' && Object.keys(resultsResponse.results).length > 0) {
          setSimulationRun(true);
        }
        
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

  // Handle component selection changes
  const handleComponentSelectionChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    const selectedValues = typeof value === 'string' ? value.split(',') : value;
    setSelectedComponents(selectedValues);
    
    // Initialize weights for selected components
    const initialWeights: {[key: string]: number} = {};
    selectedValues.forEach(component => {
      initialWeights[component] = weights[component] || (1 / selectedValues.length);
    });
    setWeights(initialWeights);
  };

  // Handle weight changes
  const handleWeightChange = (component: string, value: number) => {
    setWeights({
      ...weights,
      [component]: value
    });
  };

  // Normalize weights to sum to 1
  const normalizeWeights = () => {
    const totalWeight = Object.values(weights).reduce((sum, weight) => sum + weight, 0);
    
    if (totalWeight === 0) return;
    
    const normalizedWeights: {[key: string]: number} = {};
    Object.entries(weights).forEach(([component, weight]) => {
      normalizedWeights[component] = weight / totalWeight;
    });
    
    setWeights(normalizedWeights);
  };

  // Handle objective change
  const handleObjectiveChange = (event: SelectChangeEvent) => {
    setObjective(event.target.value);
  };

  // Analyze portfolio
  const analyzePortfolio = async () => {
    if (selectedComponents.length === 0) {
      setAlert({
        type: 'error',
        message: 'Please select at least one component for the portfolio.'
      });
      return;
    }
    
    // Normalize weights before analysis
    normalizeWeights();
    
    setAnalyzing(true);
    setAlert(null);
    
    try {
      const response = await portfolioService.analyzePortfolio({
        weights: weights
      });
      
      if (response.status === 'success') {
        setPortfolioResults(response);
        setAlert({
          type: 'success',
          message: 'Portfolio analysis completed successfully!'
        });
      } else {
        setAlert({
          type: 'error',
          message: 'Portfolio analysis failed. Please check the configuration and try again.'
        });
      }
    } catch (error) {
      console.error('Error analyzing portfolio:', error);
      setAlert({
        type: 'error',
        message: 'An error occurred while analyzing the portfolio.'
      });
    } finally {
      setAnalyzing(false);
    }
  };

  // Optimize portfolio
  const optimizePortfolio = async () => {
    if (selectedComponents.length === 0) {
      setAlert({
        type: 'error',
        message: 'Please select at least one component for the portfolio.'
      });
      return;
    }
    
    setOptimizing(true);
    setAlert(null);
    
    try {
      const response = await portfolioService.optimizePortfolio({
        components: selectedComponents,
        objective: objective
      });
      
      if (response.status === 'success') {
        setOptimizationResults(response);
        setWeights(response.weights);
        setAlert({
          type: 'success',
          message: 'Portfolio optimization completed successfully!'
        });
      } else {
        setAlert({
          type: 'error',
          message: 'Portfolio optimization failed. Please check the configuration and try again.'
        });
      }
    } catch (error) {
      console.error('Error optimizing portfolio:', error);
      setAlert({
        type: 'error',
        message: 'An error occurred while optimizing the portfolio.'
      });
    } finally {
      setOptimizing(false);
    }
  };

  // Prepare data for pie chart
  const preparePieChartData = () => {
    return Object.entries(weights).map(([name, value]) => ({
      name,
      value
    }));
  };

  // Prepare data for risk contribution chart
  const prepareRiskContributionData = () => {
    if (!portfolioResults || !portfolioResults.risk_contributions) return [];
    
    return Object.entries(portfolioResults.risk_contributions).map(([name, value]: [string, any]) => ({
      name,
      value
    }));
  };

  const pieChartData = preparePieChartData();
  const riskContributionData = prepareRiskContributionData();

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!simulationRun) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Portfolio
        </Typography>
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="h6">
            No simulation data available. Please run a simulation first.
          </Typography>
          <Button 
            variant="contained" 
            color="primary" 
            sx={{ mt: 2 }}
            onClick={() => window.location.href = '/simulation'}
          >
            Go to Simulation
          </Button>
        </Paper>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Portfolio
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
        {/* Portfolio Configuration */}
        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardHeader title="Portfolio Configuration" />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel id="component-selection-label">Select Portfolio Components</InputLabel>
                    <Select
                      labelId="component-selection-label"
                      id="component-selection"
                      multiple
                      value={selectedComponents}
                      onChange={handleComponentSelectionChange}
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
                
                {selectedComponents.length > 0 && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle1" gutterBottom>
                      Component Weights
                    </Typography>
                    {selectedComponents.map((component) => (
                      <Box key={component} sx={{ mb: 2 }}>
                        <Typography gutterBottom>
                          {component}: {(weights[component] * 100).toFixed(1)}%
                        </Typography>
                        <Slider
                          value={weights[component] || 0}
                          onChange={(_, value) => handleWeightChange(component, value as number)}
                          step={0.01}
                          min={0}
                          max={1}
                          valueLabelDisplay="auto"
                          valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                        />
                      </Box>
                    ))}
                    
                    <Button
                      variant="outlined"
                      color="primary"
                      onClick={normalizeWeights}
                      sx={{ mt: 1 }}
                    >
                      Normalize Weights
                    </Button>
                  </Grid>
                )}
                
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="primary"
                    fullWidth
                    onClick={analyzePortfolio}
                    disabled={analyzing || selectedComponents.length === 0}
                    sx={{ mt: 2 }}
                  >
                    {analyzing ? <CircularProgress size={24} /> : 'Analyze Portfolio'}
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* Portfolio Optimization */}
        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardHeader title="Portfolio Optimization" />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel id="objective-label">Optimization Objective</InputLabel>
                    <Select
                      labelId="objective-label"
                      id="objective"
                      value={objective}
                      label="Optimization Objective"
                      onChange={handleObjectiveChange}
                    >
                      <MenuItem value="min_variance">Minimize Variance</MenuItem>
                      <MenuItem value="max_sharpe">Maximize Sharpe Ratio</MenuItem>
                      <MenuItem value="min_var">Minimize Value at Risk (VaR)</MenuItem>
                      <MenuItem value="min_es">Minimize Expected Shortfall (ES)</MenuItem>
                      <MenuItem value="max_return">Maximize Expected Return</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="secondary"
                    fullWidth
                    onClick={optimizePortfolio}
                    disabled={optimizing || selectedComponents.length === 0}
                    sx={{ mt: 2 }}
                  >
                    {optimizing ? <CircularProgress size={24} /> : 'Optimize Portfolio'}
                  </Button>
                </Grid>
                
                {optimizationResults && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle1" gutterBottom>
                      Optimization Results
                    </Typography>
                    <TableContainer component={Paper} elevation={0}>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Component</TableCell>
                            <TableCell align="right">Optimal Weight</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(optimizationResults.weights).map(([component, weight]: [string, any]) => (
                            <TableRow key={component}>
                              <TableCell>{component}</TableCell>
                              <TableCell align="right">{(weight * 100).toFixed(2)}%</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* Portfolio Composition */}
        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardHeader 
              title="Portfolio Composition" 
              subheader="Current allocation of portfolio components"
            />
            <CardContent sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
              {selectedComponents.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={pieChartData}
                      cx="50%"
                      cy="50%"
                      labelLine={true}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {pieChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(2)}%`} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <Typography variant="body1">
                  Select portfolio components to view composition
                </Typography>
              )}
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* Risk Contribution */}
        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardHeader 
              title="Risk Contribution" 
              subheader="Contribution to portfolio risk by component"
            />
            <CardContent sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
              {portfolioResults && portfolioResults.risk_contributions ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={riskContributionData}
                      cx="50%"
                      cy="50%"
                      labelLine={true}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {riskContributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(2)}%`} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <Typography variant="body1">
                  Analyze portfolio to view risk contribution
                </Typography>
              )}
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* Portfolio Metrics */}
        {portfolioResults && (
          <Grid item xs={12}>
            <StyledCard>
              <CardHeader 
                title="Portfolio Metrics" 
                subheader="Key risk and performance metrics for the portfolio"
              />
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="primary" gutterBottom>
                        Expected Return
                      </Typography>
                      <Typography variant="h4">
                        {portfolioResults.metrics?.portfolio?.mean?.toFixed(2) || 'N/A'}
                      </Typography>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="primary" gutterBottom>
                        Volatility
                      </Typography>
                      <Typography variant="h4">
                        {portfolioResults.metrics?.portfolio?.std?.toFixed(2) || 'N/A'}
                      </Typography>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="primary" gutterBottom>
                        Sharpe Ratio
                      </Typography>
                      <Typography variant="h4">
                        {portfolioResults.metrics?.portfolio?.sharpe_ratio?.toFixed(2) || 'N/A'}
                      </Typography>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="primary" gutterBottom>
                        Diversification Benefit
                      </Typography>
                      <Typography variant="h4">
                        {(portfolioResults.diversification_benefit * 100)?.toFixed(1) + '%' || 'N/A'}
                      </Typography>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="h6" gutterBottom>
                        Value at Risk (VaR)
                      </Typography>
                      <Divider sx={{ mb: 2 }} />
                      <Grid container spacing={1}>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">VaR (95%):</Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2">{portfolioResults.metrics?.portfolio?.var_95?.toFixed(2) || 'N/A'}</Typography>
                        </Grid>
                        
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">VaR (99%):</Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2">{portfolioResults.metrics?.portfolio?.var_99?.toFixed(2) || 'N/A'}</Typography>
                        </Grid>
                      </Grid>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="h6" gutterBottom>
                        Expected Shortfall (ES)
                      </Typography>
                      <Divider sx={{ mb: 2 }} />
                      <Grid container spacing={1}>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">ES (95%):</Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2">{portfolioResults.metrics?.portfolio?.es_95?.toFixed(2) || 'N/A'}</Typography>
                        </Grid>
                        
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">ES (99%):</Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2">{portfolioResults.metrics?.portfolio?.es_99?.toFixed(2) || 'N/A'}</Typography>
                        </Grid>
                      </Grid>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="h6" gutterBottom>
                        Additional Metrics
                      </Typography>
                      <Divider sx={{ mb: 2 }} />
                      <Grid container spacing={1}>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Sortino Ratio:</Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2">{portfolioResults.metrics?.portfolio?.sortino_ratio?.toFixed(2) || 'N/A'}</Typography>
                        </Grid>
                        
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Max Drawdown:</Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2">{(portfolioResults.metrics?.portfolio?.max_drawdown * 100)?.toFixed(2) + '%' || 'N/A'}</Typography>
                        </Grid>
                      </Grid>
                    </Paper>
                  </Grid>
                </Grid>
              </CardContent>
            </StyledCard>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default Portfolio;
