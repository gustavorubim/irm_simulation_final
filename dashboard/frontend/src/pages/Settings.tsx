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
  TextField,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  Snackbar,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { simulationService } from '../services/api';

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

const Settings: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [config, setConfig] = useState<any>({
    num_scenarios: 1000,
    time_horizon: 3.0,
    time_steps_per_year: 12,
    rate_tenors: [],
    initial_rates: {},
    volatility_params: {},
    mean_reversion_params: {}
  });
  const [darkMode, setDarkMode] = useState<boolean>(true);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [refreshInterval, setRefreshInterval] = useState<number>(60);
  const [snackbar, setSnackbar] = useState<{open: boolean, message: string, severity: 'success' | 'error' | 'info'}>({
    open: false,
    message: '',
    severity: 'info'
  });

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setLoading(true);
        
        // Fetch simulation configuration
        const configResponse = await simulationService.getConfig();
        setConfig(configResponse);
        
        setLoading(false);
      } catch (error) {
        console.error('Error fetching configuration:', error);
        setSnackbar({
          open: true,
          message: 'Failed to fetch configuration. Please refresh the page.',
          severity: 'error'
        });
        setLoading(false);
      }
    };
    
    fetchConfig();
  }, []);

  // Handle config changes
  const handleConfigChange = (field: string, value: any) => {
    setConfig({
      ...config,
      [field]: value
    });
  };

  // Handle rate parameter changes
  const handleRateParamChange = (paramType: string, tenor: string, value: number) => {
    const updatedParams = { ...config[paramType] };
    updatedParams[tenor] = value;
    
    setConfig({
      ...config,
      [paramType]: updatedParams
    });
  };

  // Save configuration
  const saveConfiguration = async () => {
    setSaving(true);
    
    try {
      const response = await simulationService.updateConfig(config);
      
      if (response.status === 'success') {
        setSnackbar({
          open: true,
          message: 'Configuration saved successfully!',
          severity: 'success'
        });
      } else {
        setSnackbar({
          open: true,
          message: 'Failed to save configuration.',
          severity: 'error'
        });
      }
    } catch (error) {
      console.error('Error saving configuration:', error);
      setSnackbar({
        open: true,
        message: 'An error occurred while saving configuration.',
        severity: 'error'
      });
    } finally {
      setSaving(false);
    }
  };

  // Reset configuration to defaults
  const resetConfiguration = async () => {
    try {
      setLoading(true);
      
      // Fetch default configuration
      const configResponse = await simulationService.getConfig();
      setConfig(configResponse);
      
      setSnackbar({
        open: true,
        message: 'Configuration reset to defaults.',
        severity: 'info'
      });
      
      setLoading(false);
    } catch (error) {
      console.error('Error resetting configuration:', error);
      setSnackbar({
        open: true,
        message: 'Failed to reset configuration.',
        severity: 'error'
      });
      setLoading(false);
    }
  };

  // Export configuration
  const exportConfiguration = () => {
    const configJson = JSON.stringify(config, null, 2);
    const blob = new Blob([configJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'irm_simulation_config.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    setSnackbar({
      open: true,
      message: 'Configuration exported successfully!',
      severity: 'success'
    });
  };

  // Import configuration
  const importConfiguration = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importedConfig = JSON.parse(e.target?.result as string);
        setConfig(importedConfig);
        
        setSnackbar({
          open: true,
          message: 'Configuration imported successfully!',
          severity: 'success'
        });
      } catch (error) {
        console.error('Error parsing imported configuration:', error);
        setSnackbar({
          open: true,
          message: 'Failed to import configuration. Invalid format.',
          severity: 'error'
        });
      }
    };
    
    reader.readAsText(file);
  };

  // Handle snackbar close
  const handleSnackbarClose = () => {
    setSnackbar({
      ...snackbar,
      open: false
    });
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={6000} 
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleSnackbarClose} 
          severity={snackbar.severity} 
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
      
      <Grid container spacing={3}>
        {/* Simulation Settings */}
        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardHeader 
              title="Simulation Settings" 
              subheader="Configure simulation parameters"
              action={
                <IconButton onClick={saveConfiguration} disabled={saving}>
                  <SaveIcon />
                </IconButton>
              }
            />
            <CardContent>
              <Grid container spacing={2}>
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
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Time Steps per Year"
                    type="number"
                    value={config.time_steps_per_year}
                    onChange={(e) => handleConfigChange('time_steps_per_year', parseInt(e.target.value))}
                    InputProps={{ inputProps: { min: 4, max: 52, step: 4 } }}
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle1" gutterBottom>
                    Initial Rates
                  </Typography>
                  <Grid container spacing={2}>
                    {config.rate_tenors && config.rate_tenors.map((tenor: string) => (
                      <Grid item xs={12} sm={6} key={`initial-${tenor}`}>
                        <TextField
                          fullWidth
                          label={`${tenor} Rate`}
                          type="number"
                          value={config.initial_rates[tenor]}
                          onChange={(e) => handleRateParamChange('initial_rates', tenor, parseFloat(e.target.value))}
                          InputProps={{ inputProps: { min: 0, max: 0.2, step: 0.001 } }}
                        />
                      </Grid>
                    ))}
                  </Grid>
                </Grid>
                
                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle1" gutterBottom>
                    Volatility Parameters
                  </Typography>
                  <Grid container spacing={2}>
                    {config.rate_tenors && config.rate_tenors.map((tenor: string) => (
                      <Grid item xs={12} sm={6} key={`vol-${tenor}`}>
                        <TextField
                          fullWidth
                          label={`${tenor} Volatility`}
                          type="number"
                          value={config.volatility_params[tenor]}
                          onChange={(e) => handleRateParamChange('volatility_params', tenor, parseFloat(e.target.value))}
                          InputProps={{ inputProps: { min: 0.001, max: 0.05, step: 0.001 } }}
                        />
                      </Grid>
                    ))}
                  </Grid>
                </Grid>
                
                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle1" gutterBottom>
                    Mean Reversion Parameters
                  </Typography>
                  <Grid container spacing={2}>
                    {config.rate_tenors && config.rate_tenors.map((tenor: string) => (
                      <Grid item xs={12} sm={6} key={`mr-${tenor}`}>
                        <TextField
                          fullWidth
                          label={`${tenor} Mean Reversion`}
                          type="number"
                          value={config.mean_reversion_params[tenor]}
                          onChange={(e) => handleRateParamChange('mean_reversion_params', tenor, parseFloat(e.target.value))}
                          InputProps={{ inputProps: { min: 0.01, max: 0.5, step: 0.01 } }}
                        />
                      </Grid>
                    ))}
                  </Grid>
                </Grid>
                
                <Grid item xs={12}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                    <Button
                      variant="outlined"
                      color="primary"
                      startIcon={<RefreshIcon />}
                      onClick={resetConfiguration}
                    >
                      Reset to Defaults
                    </Button>
                    
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={<SaveIcon />}
                      onClick={saveConfiguration}
                      disabled={saving}
                    >
                      {saving ? 'Saving...' : 'Save Configuration'}
                    </Button>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* UI Settings */}
        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardHeader 
              title="UI Settings" 
              subheader="Configure user interface preferences"
            />
            <CardContent>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <SettingsIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Dark Mode" 
                    secondary="Enable dark theme for the application"
                  />
                  <Switch
                    edge="end"
                    checked={darkMode}
                    onChange={(e) => setDarkMode(e.target.checked)}
                  />
                </ListItem>
                
                <Divider variant="inset" component="li" />
                
                <ListItem>
                  <ListItemIcon>
                    <RefreshIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Auto Refresh" 
                    secondary="Automatically refresh data at specified interval"
                  />
                  <Switch
                    edge="end"
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                  />
                </ListItem>
                
                {autoRefresh && (
                  <ListItem>
                    <TextField
                      fullWidth
                      label="Refresh Interval (seconds)"
                      type="number"
                      value={refreshInterval}
                      onChange={(e) => setRefreshInterval(parseInt(e.target.value))}
                      InputProps={{ inputProps: { min: 10, max: 3600 } }}
                    />
                  </ListItem>
                )}
                
                <Divider variant="inset" component="li" />
                
                <ListItem>
                  <ListItemIcon>
                    <DownloadIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Export Configuration" 
                    secondary="Save current configuration to a file"
                  />
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<DownloadIcon />}
                    onClick={exportConfiguration}
                  >
                    Export
                  </Button>
                </ListItem>
                
                <Divider variant="inset" component="li" />
                
                <ListItem>
                  <ListItemIcon>
                    <UploadIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Import Configuration" 
                    secondary="Load configuration from a file"
                  />
                  <Button
                    variant="outlined"
                    size="small"
                    component="label"
                    startIcon={<UploadIcon />}
                  >
                    Import
                    <input
                      type="file"
                      accept=".json"
                      hidden
                      onChange={importConfiguration}
                    />
                  </Button>
                </ListItem>
              </List>
            </CardContent>
          </StyledCard>
        </Grid>
        
        {/* About */}
        <Grid item xs={12}>
          <StyledCard>
            <CardHeader 
              title="About" 
              subheader="EVE and EVS Simulation System"
            />
            <CardContent>
              <Typography variant="body1" paragraph>
                This application provides a comprehensive stochastic simulation system for Economic Value of Equity (EVE) and 
                Economic Value Sensitivity (EVS) in the context of interest rate risk management.
              </Typography>
              
              <Typography variant="body1" paragraph>
                The system includes:
              </Typography>
              
              <List>
                <ListItem>
                  <ListItemIcon>
                    <SettingsIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Regression Models" 
                    secondary="A collection of ~20 different regression models with factor sensitivity to interest rate movements"
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <SettingsIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Interest Rate Simulation" 
                    secondary="Monte Carlo simulation using the Heath-Jarrow-Morton (HJM) model for interest rates"
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <SettingsIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Comprehensive Metrics" 
                    secondary="Calculation of various risk and performance metrics"
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <SettingsIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Interactive Dashboard" 
                    secondary="Modern, visually impressive dashboard for data visualization and analysis"
                  />
                </ListItem>
              </List>
              
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
                Version 1.0.0 | © 2025 | All Rights Reserved
              </Typography>
            </CardContent>
          </StyledCard>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Settings;
