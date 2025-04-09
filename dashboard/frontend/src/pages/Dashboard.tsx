import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Card, 
  CardContent, 
  CardHeader,
  Button,
  CircularProgress,
  Divider
} from '@mui/material';
import Grid from '@mui/material/Grid'; // Changed import style
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
  Area,
  BarChart,
  Bar
} from 'recharts';
import { simulationService, metricsService, chartsService } from '../services/api';

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

const MetricValue = styled(Typography)(({ theme }) => ({
  fontSize: '2rem',
  fontWeight: 'bold',
  marginBottom: theme.spacing(1),
}));

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(true);
  const [simulationData, setSimulationData] = useState<any>(null);
  const [metricsData, setMetricsData] = useState<any>(null);
  const [histogramChart, setHistogramChart] = useState<string>('');
  const [correlationChart, setCorrelationChart] = useState<string>('');
  const [sensitivityChart, setSensitivityChart] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Fetch simulation results
        const resultsResponse = await simulationService.getResults();
        if (resultsResponse.status === 'success') {
          setSimulationData(resultsResponse.results);
          
          // Set selected model to first model in results
          if (Object.keys(resultsResponse.results).length > 0) {
            setSelectedModel(Object.keys(resultsResponse.results)[0]);
          }
        }
        
        // Fetch metrics
        const metricsResponse = await metricsService.getMetrics();
        if (metricsResponse.status === 'success') {
          setMetricsData(metricsResponse.metrics);
        }
        
        // Fetch charts
        if (selectedModel) {
          const histogramResponse = await chartsService.getHistogram(selectedModel);
          if (histogramResponse.status === 'success') {
            setHistogramChart(histogramResponse.chart);
          }
        }
        
        const correlationResponse = await chartsService.getCorrelation();
        if (correlationResponse.status === 'success') {
          setCorrelationChart(correlationResponse.chart);
        }
        
        const sensitivityResponse = await chartsService.getSensitivity();
        if (sensitivityResponse.status === 'success') {
          setSensitivityChart(sensitivityResponse.chart);
        }
        
        setLoading(false);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setLoading(false);
      }
    };
    
    fetchData();
  }, [selectedModel]);
  
  // Prepare time series data for the selected model
  const [timeseriesData, setTimeseriesData] = useState<any[]>([]);
  
  useEffect(() => {
    const fetchTimeseriesData = async () => {
      if (selectedModel) {
        try {
          const response = await simulationService.getTimeseries(selectedModel);
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
            setTimeseriesData(formattedData);
          }
        } catch (error) {
          console.error('Error fetching timeseries data:', error);
        }
      }
    };
    
    fetchTimeseriesData();
  }, [selectedModel]);

  // Extract key metrics for the selected model
  const getModelMetrics = () => {
    if (!metricsData || !selectedModel || !metricsData[selectedModel]) {
      return null;
    }
    
    const metrics = metricsData[selectedModel];
    return {
      mean: metrics.mean?.toFixed(2) || 'N/A',
      std: metrics.std?.toFixed(2) || 'N/A',
      var95: metrics.var_95?.toFixed(2) || 'N/A',
      es95: metrics.es_95?.toFixed(2) || 'N/A',
      sharpeRatio: metrics.sharpe_ratio?.toFixed(2) || 'N/A',
      maxDrawdown: (metrics.max_drawdown * 100)?.toFixed(2) + '%' || 'N/A',
    };
  };
  
  const modelMetrics = getModelMetrics();

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!simulationData) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Dashboard
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
        Dashboard
      </Typography>
      
      {/* Key Metrics Section */}
      {modelMetrics && (
      <Grid container={true} spacing={3} sx={{ mb: 4 }}>
            <Grid item={true} xs={12} md={4} key="var95">
              <StyledCard>
                <CardHeader title="Value at Risk (95%)" />
                <CardContent>
                  <MetricValue color="error">{modelMetrics.var95}</MetricValue>
                  <Typography variant="body2" color="text.secondary">
                    Maximum potential loss at 95% confidence level
                  </Typography>
                </CardContent>
              </StyledCard>
            </Grid>
            
            <Grid item={true} xs={12} md={4} key="es95">
              <StyledCard>
                <CardHeader title="Expected Shortfall (95%)" />
                <CardContent>
                  <MetricValue color="error">{modelMetrics.es95}</MetricValue>
                  <Typography variant="body2" color="text.secondary">
                    Average loss in the worst 5% of scenarios
                  </Typography>
                </CardContent>
              </StyledCard>
            </Grid>
            
            <Grid item={true} xs={12} md={4} key="sharpeRatio">
              <StyledCard>
                <CardHeader title="Sharpe Ratio" />
                <CardContent>
                  <MetricValue color="primary">{modelMetrics.sharpeRatio}</MetricValue>
                  <Typography variant="body2" color="text.secondary">
                    Risk-adjusted return measure
                  </Typography>
                </CardContent>
              </StyledCard>
            </Grid>
      </Grid>
      )}
      
      {/* Charts Section */}
      <Grid container spacing={3}>
        {/* Time Series Chart */}
        <Grid item xs={12} lg={6}>
          <StyledCard>
            <CardHeader 
              title={`${selectedModel} Time Series`} 
              subheader="Projected values over time"
            />
            <CardContent sx={{ flexGrow: 1 }}>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={timeseriesData}>
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
        
        {/* Distribution Histogram */}
        <Grid item xs={12} lg={6}>
          <StyledCard>
            <CardHeader 
              title={`${selectedModel} Distribution`} 
              subheader="Histogram of simulation results"
            />
            <CardContent sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center' }}>
              {histogramChart ? (
                <img src={histogramChart} alt="Distribution Histogram" style={{ maxWidth: '100%', maxHeight: 300 }} />
              ) : (
                <Typography variant="body1">No histogram data available</Typography>
              )}
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* Correlation Matrix */}
        <Grid item xs={12} lg={6}>
          <StyledCard>
            <CardHeader 
              title="Correlation Matrix" 
              subheader="Correlation between different models"
            />
            <CardContent sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center' }}>
              {correlationChart ? (
                <img src={correlationChart} alt="Correlation Matrix" style={{ maxWidth: '100%', maxHeight: 300 }} />
              ) : (
                <Typography variant="body1">No correlation data available</Typography>
              )}
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* Rate Sensitivity */}
        <Grid item xs={12} lg={6}>
          <StyledCard>
            <CardHeader 
              title="Rate Sensitivity" 
              subheader="Sensitivity to interest rate changes"
            />
            <CardContent sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center' }}>
              {sensitivityChart ? (
                <img src={sensitivityChart} alt="Rate Sensitivity" style={{ maxWidth: '100%', maxHeight: 300 }} />
              ) : (
                <Typography variant="body1">No sensitivity data available</Typography>
              )}
            </CardContent>
          </StyledCard>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
