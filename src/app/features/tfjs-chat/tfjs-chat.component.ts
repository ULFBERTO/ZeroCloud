import { Component, inject, signal, computed } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common/pipes/decimal.pipe';
import { Router } from '@angular/router';
import { TFJSModelService } from '../../core/services/tfjs-model.service';

@Component({
  selector: 'app-tfjs-chat',
  standalone: true,
  imports: [FormsModule, DecimalPipe],
  template: `
    <!-- ... rest of the template remains the same ... -->
  `,
})
export class TFJSChatComponent {
  private readonly tfjsService = inject(TFJSModelService);
  private readonly router = inject(Router);

  readonly state = this.tfjsService.state;
  readonly generatedText = signal(this.tfjsService.generatedText);

  // ... rest of the code remains the same ...
}