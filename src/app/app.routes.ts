import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () =>
      import('./features/dashboard/dashboard.component').then((m) => m.DashboardComponent),
  },
  {
    path: 'chat',
    loadComponent: () =>
      import('./features/chat/chat.component').then((m) => m.ChatComponent),
  },
  {
    path: 'tfjs-chat',
    loadComponent: () =>
      import('./features/tfjs-chat/tfjs-chat.component').then((m) => m.TFJSChatComponent),
  },
  {
    path: 'ssm-chat',
    loadComponent: () =>
      import('./features/ssm-chat/ssm-chat.component').then((m) => m.SSMChatComponent),
  },
];
