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
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip
} from '@mui/material';
import { styled } from '@mui/material/styles';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { modelsService } from '../services/api';

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

const Models: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(true);
  const [modelDetails, setModelDetails] = useState<any>(null);
  const [eveModels, setEveModels] = useState<string[]>([]);
  const [evsModels, setEvsModels] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        
        // Fetch model list
        const modelsResponse = await modelsService.getModels();
        setEveModels(modelsResponse.eve_models || []);
        setEvsModels(modelsResponse.evs_models || []);
        
        // Fetch model details
        const detailsResponse = await modelsService.getModelDetails();
        setModelDetails(detailsResponse);
        
        setLoading(false);
      } catch (error) {
        console.error('Error fetching models:', error);
        setError('Failed to fetch model information. Please try again later.');
        setLoading(false);
      }
    };
    
    fetchModels();
  }, []);

  // Group models by type
  const groupedModels = {
    eve: eveModels.map(model => modelDetails?.[model] || { name: model }),
    evs: evsModels.map(model => modelDetails?.[model] || { name: model })
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Models
        </Typography>
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="h6" color="error">
            {error}
          </Typography>
        </Paper>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Models
      </Typography>
      
      <Grid container spacing={3}>
        {/* EVE Models Section */}
        <Grid item xs={12}>
          <StyledCard>
            <CardHeader 
              title="Economic Value of Equity (EVE) Models" 
              subheader="Models that estimate the change in economic value of equity based on interest rate movements"
            />
            <CardContent>
              <Accordion defaultExpanded>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel1a-content"
                  id="panel1a-header"
                >
                  <Typography variant="h6">Model Overview</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <TableContainer component={Paper} elevation={0}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Model Name</TableCell>
                          <TableCell>Description</TableCell>
                          <TableCell>Type</TableCell>
                          <TableCell>R²</TableCell>
                          <TableCell>Status</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {groupedModels.eve.map((model: any) => (
                          <TableRow key={model.name}>
                            <TableCell>{model.name}</TableCell>
                            <TableCell>{model.description || 'No description available'}</TableCell>
                            <TableCell>{model.model_type || 'linear'}</TableCell>
                            <TableCell>{model.r_squared ? model.r_squared.toFixed(2) : 'N/A'}</TableCell>
                            <TableCell>
                              <Chip 
                                label={model.is_fitted ? "Fitted" : "Not Fitted"} 
                                color={model.is_fitted ? "success" : "warning"} 
                                size="small" 
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </AccordionDetails>
              </Accordion>
              
              <Accordion>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel2a-content"
                  id="panel2a-header"
                >
                  <Typography variant="h6">Rate Sensitivities</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <TableContainer component={Paper} elevation={0}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Model Name</TableCell>
                          <TableCell>SOFR Rate</TableCell>
                          <TableCell>Treasury 1Y</TableCell>
                          <TableCell>Treasury 2Y</TableCell>
                          <TableCell>Treasury 3Y</TableCell>
                          <TableCell>Treasury 5Y</TableCell>
                          <TableCell>Treasury 10Y</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {groupedModels.eve.map((model: any) => (
                          <TableRow key={model.name}>
                            <TableCell>{model.name}</TableCell>
                            <TableCell>{model.rate_sensitivities?.sofr_rate?.toFixed(2) || 'N/A'}</TableCell>
                            <TableCell>{model.rate_sensitivities?.treasury_1y?.toFixed(2) || 'N/A'}</TableCell>
                            <TableCell>{model.rate_sensitivities?.treasury_2y?.toFixed(2) || 'N/A'}</TableCell>
                            <TableCell>{model.rate_sensitivities?.treasury_3y?.toFixed(2) || 'N/A'}</TableCell>
                            <TableCell>{model.rate_sensitivities?.treasury_5y?.toFixed(2) || 'N/A'}</TableCell>
                            <TableCell>{model.rate_sensitivities?.treasury_10y?.toFixed(2) || 'N/A'}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </AccordionDetails>
              </Accordion>
              
              <Accordion>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel3a-content"
                  id="panel3a-header"
                >
                  <Typography variant="h6">Model Coefficients</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    {groupedModels.eve.map((model: any) => (
                      <Grid item xs={12} md={6} lg={4} key={model.name}>
                        <Paper sx={{ p: 2 }}>
                          <Typography variant="subtitle1" gutterBottom>{model.name}</Typography>
                          <Divider sx={{ mb: 2 }} />
                          
                          <Typography variant="body2" gutterBottom>
                            <strong>Intercept:</strong> {model.intercept?.toFixed(4) || 'N/A'}
                          </Typography>
                          
                          <Typography variant="body2" gutterBottom>
                            <strong>Coefficients:</strong>
                          </Typography>
                          
                          {model.coefficients && Object.entries(model.coefficients).map(([key, value]: [string, any]) => (
                            <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                              <Typography variant="body2" color="text.secondary">{key}:</Typography>
                              <Typography variant="body2">{value.toFixed(4)}</Typography>
                            </Box>
                          ))}
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* EVS Models Section */}
        <Grid item xs={12}>
          <StyledCard>
            <CardHeader 
              title="Economic Value Sensitivity (EVS) Models" 
              subheader="Models that estimate the sensitivity of economic value to interest rate movements"
            />
            <CardContent>
              <Accordion defaultExpanded>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel4a-content"
                  id="panel4a-header"
                >
                  <Typography variant="h6">Model Overview</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <TableContainer component={Paper} elevation={0}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Model Name</TableCell>
                          <TableCell>Description</TableCell>
                          <TableCell>Type</TableCell>
                          <TableCell>Duration Based</TableCell>
                          <TableCell>R²</TableCell>
                          <TableCell>Status</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {groupedModels.evs.map((model: any) => (
                          <TableRow key={model.name}>
                            <TableCell>{model.name}</TableCell>
                            <TableCell>{model.description || 'No description available'}</TableCell>
                            <TableCell>{model.model_type || 'linear'}</TableCell>
                            <TableCell>{model.duration_based ? 'Yes' : 'No'}</TableCell>
                            <TableCell>{model.r_squared ? model.r_squared.toFixed(2) : 'N/A'}</TableCell>
                            <TableCell>
                              <Chip 
                                label={model.is_fitted ? "Fitted" : "Not Fitted"} 
                                color={model.is_fitted ? "success" : "warning"} 
                                size="small" 
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </AccordionDetails>
              </Accordion>
              
              <Accordion>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel5a-content"
                  id="panel5a-header"
                >
                  <Typography variant="h6">Durations & Convexities</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <TableContainer component={Paper} elevation={0}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Model Name</TableCell>
                          <TableCell colSpan={2}>SOFR Rate</TableCell>
                          <TableCell colSpan={2}>Treasury 1Y</TableCell>
                          <TableCell colSpan={2}>Treasury 5Y</TableCell>
                          <TableCell colSpan={2}>Treasury 10Y</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell></TableCell>
                          <TableCell>Duration</TableCell>
                          <TableCell>Convexity</TableCell>
                          <TableCell>Duration</TableCell>
                          <TableCell>Convexity</TableCell>
                          <TableCell>Duration</TableCell>
                          <TableCell>Convexity</TableCell>
                          <TableCell>Duration</TableCell>
                          <TableCell>Convexity</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {groupedModels.evs
                          .filter((model: any) => model.duration_based)
                          .map((model: any) => (
                          <TableRow key={model.name}>
                            <TableCell>{model.name}</TableCell>
                            <TableCell>{model.durations?.sofr_rate?.toFixed(2) || 'N/A'}</TableCell>
                            <TableCell>{model.convexities?.sofr_rate?.toFixed(2) || 'N/A'}</TableCell>
                            <TableCell>{model.durations?.treasury_1y?.toFixed(2) || 'N/A'}</TableCell>
                            <TableCell>{model.convexities?.treasury_1y?.toFixed(2) || 'N/A'}</TableCell>
                            <TableCell>{model.durations?.treasury_5y?.toFixed(2) || 'N/A'}</TableCell>
                            <TableCell>{model.convexities?.treasury_5y?.toFixed(2) || 'N/A'}</TableCell>
                            <TableCell>{model.durations?.treasury_10y?.toFixed(2) || 'N/A'}</TableCell>
                            <TableCell>{model.convexities?.treasury_10y?.toFixed(2) || 'N/A'}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </AccordionDetails>
              </Accordion>
              
              <Accordion>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel6a-content"
                  id="panel6a-header"
                >
                  <Typography variant="h6">Model Coefficients</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    {groupedModels.evs.map((model: any) => (
                      <Grid item xs={12} md={6} lg={4} key={model.name}>
                        <Paper sx={{ p: 2 }}>
                          <Typography variant="subtitle1" gutterBottom>{model.name}</Typography>
                          <Divider sx={{ mb: 2 }} />
                          
                          <Typography variant="body2" gutterBottom>
                            <strong>Intercept:</strong> {model.intercept?.toFixed(4) || 'N/A'}
                          </Typography>
                          
                          <Typography variant="body2" gutterBottom>
                            <strong>Coefficients:</strong>
                          </Typography>
                          
                          {model.coefficients && Object.entries(model.coefficients).map(([key, value]: [string, any]) => (
                            <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                              <Typography variant="body2" color="text.secondary">{key}:</Typography>
                              <Typography variant="body2">{value.toFixed(4)}</Typography>
                            </Box>
                          ))}
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* Model Documentation */}
        <Grid item xs={12}>
          <StyledCard>
            <CardHeader 
              title="Model Documentation" 
              subheader="Technical details about the models"
            />
            <CardContent>
              <Typography variant="body1" paragraph>
                The EVE and EVS models in this system are regression-based models that capture the relationship between interest rates and economic value metrics.
              </Typography>
              
              <Typography variant="h6" gutterBottom>
                EVE Models
              </Typography>
              <Typography variant="body1" paragraph>
                Economic Value of Equity (EVE) models estimate the present value of expected future cash flows of assets minus liabilities. 
                These models capture how the economic value changes in response to interest rate movements and other factors.
              </Typography>
              <Typography variant="body1" paragraph>
                The models use either linear or log-linear regression approaches, with coefficients representing the sensitivity to various factors.
                The primary factors are interest rates at different tenors (SOFR, Treasury 1Y, 2Y, 3Y, 5Y, 10Y), but some models also include
                additional factors like credit spreads, prepayment speeds, and other market variables.
              </Typography>
              
              <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                EVS Models
              </Typography>
              <Typography variant="body1" paragraph>
                Economic Value Sensitivity (EVS) models measure the sensitivity of economic value to changes in interest rates.
                These models can be regression-based or duration-based.
              </Typography>
              <Typography variant="body1" paragraph>
                Duration-based models use the concepts of duration and convexity from fixed income analysis to estimate sensitivity.
                Duration represents the first-order (linear) sensitivity, while convexity captures the second-order (non-linear) effects.
                Higher duration values indicate greater sensitivity to interest rate changes.
              </Typography>
              
              <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                Model Interpretation
              </Typography>
              <Typography variant="body1" paragraph>
                <strong>Coefficients:</strong> Represent the change in the dependent variable (EVE or EVS) for a one-unit change in the independent variable (e.g., interest rate).
                Negative coefficients for interest rates in EVE models indicate that the economic value decreases when rates increase.
              </Typography>
              <Typography variant="body1" paragraph>
                <strong>R²:</strong> Coefficient of determination, indicating how well the model fits the data. Values closer to 1 indicate better fit.
              </Typography>
              <Typography variant="body1" paragraph>
                <strong>Duration:</strong> Measures the sensitivity of economic value to interest rate changes. Higher duration means greater sensitivity.
              </Typography>
              <Typography variant="body1" paragraph>
                <strong>Convexity:</strong> Measures the curvature of the relationship between economic value and interest rates. Positive convexity is beneficial as it means the value increases more when rates fall than it decreases when rates rise by the same amount.
              </Typography>
            </CardContent>
          </StyledCard>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Models;
