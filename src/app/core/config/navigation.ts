import { IconName } from '../../shared/components/icon.component';

export interface NavItem {
  label: string;
  path: string;
  icon: IconName;
  exact?: boolean;
}

export interface ExternalNavItem {
  label: string;
  href: string;
  icon: IconName;
  external: true;
}

export const PRIMARY_NAV: NavItem[] = [
  { label: 'Inicio', path: '/', icon: 'home', exact: true },
  { label: 'Playground', path: '/playground', icon: 'flask' },
  { label: 'Modelos', path: '/models', icon: 'layers' },
  { label: 'Uso', path: '/usage', icon: 'chart' },
  { label: 'Documentación', path: '/docs', icon: 'book' },
];

export const RESOURCE_NAV: ExternalNavItem[] = [
  {
    label: 'GitHub',
    href: 'https://github.com/ULFBERTO/ZeroCloud',
    icon: 'github',
    external: true,
  },
  {
    label: 'Hugging Face',
    href: 'https://huggingface.co/ULFBERTO/OxideLLM_TK_SSM_V1',
    icon: 'external',
    external: true,
  },
];
