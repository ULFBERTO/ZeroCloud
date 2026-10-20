import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { IconComponent } from '../../shared/components';

interface DocSection {
  id: string;
  label: string;
  index: string;
}

@Component({
  selector: 'app-docs',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink, IconComponent],
  templateUrl: './docs.component.html',
})
export class DocsComponent {
  readonly sections: DocSection[] = [
    { id: 'overview', label: 'Visión general', index: '01' },
    { id: 'quickstart', label: 'Inicio rápido', index: '02' },
    { id: 'models', label: 'Modelos', index: '03' },
    { id: 'runtimes', label: 'Runtimes', index: '04' },
    { id: 'p2p', label: 'Clúster P2P', index: '05' },
    { id: 'privacy', label: 'Privacidad', index: '06' },
  ];
}
