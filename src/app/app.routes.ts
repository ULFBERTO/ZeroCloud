import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () =>
      import('./layout/app-shell.component').then((m) => m.AppShellComponent),
    children: [
      {
        path: '',
        title: 'ZeroCloud — Inferencia local en tu navegador',
        loadComponent: () => import('./features/home/home.component').then((m) => m.HomeComponent),
      },
      {
        path: 'playground',
        title: 'Playground — ZeroCloud',
        loadComponent: () =>
          import('./features/chat/chat.component').then((m) => m.ChatComponent),
      },
      {
        path: 'models',
        title: 'Modelos — ZeroCloud',
        loadComponent: () =>
          import('./features/models/models.component').then((m) => m.ModelsComponent),
      },
      {
        path: 'usage',
        title: 'Uso — ZeroCloud',
        loadComponent: () =>
          import('./features/usage/usage.component').then((m) => m.UsageComponent),
      },
      {
        path: 'docs',
        title: 'Documentación — ZeroCloud',
        loadComponent: () =>
          import('./features/docs/docs.component').then((m) => m.DocsComponent),
      },
      {
        path: 'ssm-chat',
        title: 'OxideLLM SSM — ZeroCloud',
        loadComponent: () =>
          import('./features/ssm-chat/ssm-chat.component').then((m) => m.SSMChatComponent),
      },
      {
        path: 'tfjs-chat',
        title: 'OxideLLM TFJS — ZeroCloud',
        loadComponent: () =>
          import('./features/tfjs-chat/tfjs-chat.component').then((m) => m.TFJSChatComponent),
      },
      {
        path: 'chat',
        pathMatch: 'full',
        redirectTo: 'playground',
      },
    ],
  },
  {
    path: '**',
    redirectTo: '',
  },
];
