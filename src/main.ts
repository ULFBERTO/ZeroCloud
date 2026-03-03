import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { App } from './app/app';
import { initFromConfig } from 'healcode'; // assuming it should be 'healcode'

initFromConfig();

bootstrapApplication(App, appConfig)
  .catch((err) => console.error(err));
